"""ImageJ macro language subset and the Python scripting facade (Plugins > Macros > Run...,
Install..., the Recorder's "Create", Edit > Options > Startup...).

The language follows ImageJ 1.54 ``ij/macro`` (public domain, NIH): numbers are doubles,
booleans are 1/0, strings concatenate with ``+`` (numbers formatted like ImageJ: integers
without decimals, otherwise 4 significant decimals), ``==`` on strings ignores case, ``^`` is
bitwise XOR; ``var`` declares globals visible in functions; ``function f(a, b) {...}``;
``macro "Name" {...}`` blocks (installable); ``if/else``, ``for``, ``while``, ``do..while``,
``break``, ``continue``, ``return``; arrays (``newArray``, ``a[i]``, ``a.length``, ``Array.*``);
string methods (``s.length``, ``s.contains()``...). Functions with output arguments
(``getDimensions(w, h, c, z, t)``, ``getSelectionBounds``, ``getStatistics`` ...) assign the
caller's variables as ImageJ does.

Built-ins cover what the Recorder writes and the commonly used image functions: ``run``,
``open``, ``close``, ``selectWindow``/``selectImage``, ``newImage``, ``saveAs``, ``rename``,
``getTitle``, ``nImages``, ``nSlices``, ``setSlice``, ``Stack.*``, ``make*`` selections,
``selectionType``, ``getSelectionBounds``/``Coordinates``, ``roiManager``, ``setThreshold``,
``setAutoThreshold`` (through the Process package), ``getValue``, ``getStatistics``,
``getPixel``/``setPixel``, results functions, ``print``, string/math functions, ``File.*``,
``Roi.*``, ``Overlay.*``, ``Property.*`` and the dialog functions (``getNumber``,
``Dialog.*``; answered by the UI hook or with their defaults when headless).

Python scripts run with an ``ij`` object (:class:`ScriptAPI`) whose methods are the same
built-ins (``ij.run("Gaussian Blur...", sigma=2)``, ``ij.makeRectangle(1, 2, 3, 4)``,
``ij.getValue("Mean")``), plus ``session``, ``image`` and numpy as ``np``.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


class MacroError(Exception):
    def __init__(self, message: str, line: int = 0):
        super().__init__(f"{message} in line {line}" if line else message)
        self.message = message
        self.line = line


class MacroExit(Exception):
    """``exit`` / ``exit("message")``."""


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


# ---------------------------------------------------------------------------------- tokens
@dataclass
class Tok:
    kind: str        # "num" "str" "id" "op" "eof"
    value: Any
    line: int


_OPS = ("<<=", ">>=", "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=",
        "<<", ">>", "+", "-", "*", "/", "%", "^", "&", "|", "~", "!", "<", ">", "=", "(", ")",
        "[", "]", "{", "}", ",", ";", ".", "?", ":")
KEYWORDS = {"var", "function", "return", "if", "else", "for", "while", "do", "macro", "break",
            "continue", "requires"}


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r\f":
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += src.count("\n", i, j)
            i = j
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = re.compile(r"0[xX][0-9a-fA-F]+|(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?").match(src, i)
            text = m.group(0)
            val = float(int(text, 16)) if text[:2].lower() == "0x" else float(text)
            toks.append(Tok("num", val, line))
            i = m.end()
            continue
        if c.isalpha() or c == "_":
            m = re.compile(r"[A-Za-z_][A-Za-z0-9_]*").match(src, i)
            toks.append(Tok("id", m.group(0), line))
            i = m.end()
            continue
        if c in "\"'":
            q = c
            j = i + 1
            out = []
            while j < n and src[j] != q:
                ch = src[j]
                if ch == "\\" and j + 1 < n:
                    nx = src[j + 1]
                    if nx == "n":
                        out.append("\n")
                    elif nx == "t":
                        out.append("\t")
                    elif nx == "r":
                        out.append("\r")
                    elif nx == "u" and j + 5 < n:
                        out.append(chr(int(src[j + 2:j + 6], 16)))
                        j += 4
                    else:
                        out.append(nx)
                    j += 2
                    continue
                if ch == "\n":
                    line += 1
                out.append(ch)
                j += 1
            if j >= n:
                raise MacroError("Unterminated string", line)
            toks.append(Tok("str", "".join(out), line))
            i = j + 1
            continue
        for op in _OPS:
            if src.startswith(op, i):
                toks.append(Tok("op", op, line))
                i += len(op)
                break
        else:
            raise MacroError(f"Invalid character '{c}'", line)
    toks.append(Tok("eof", None, line))
    return toks


# ---------------------------------------------------------------------------------- parser
@dataclass
class FuncDef:
    name: str
    params: list[str]
    body: list
    line: int = 0


@dataclass
class Program:
    body: list = field(default_factory=list)
    functions: dict = field(default_factory=dict)
    macros: list = field(default_factory=list)       # [(name, body)]


NAMESPACES = {"Stack", "Roi", "Overlay", "File", "Array", "String", "Math", "IJ", "Table",
              "Property", "List", "Color", "Image", "RoiManager", "Dialog", "Plot", "Fit"}


class Parser:
    def __init__(self, toks: list[Tok]):
        self.t = toks
        self.p = 0

    def peek(self, k: int = 0) -> Tok:
        return self.t[min(self.p + k, len(self.t) - 1)]

    def next(self) -> Tok:
        tok = self.t[self.p]
        self.p += 1
        return tok

    def is_op(self, v: str, k: int = 0) -> bool:
        tok = self.peek(k)
        return tok.kind == "op" and tok.value == v

    def is_kw(self, v: str) -> bool:
        tok = self.peek()
        return tok.kind == "id" and tok.value == v

    def expect(self, v: str) -> Tok:
        tok = self.next()
        if tok.kind != "op" or tok.value != v:
            got = tok.value if tok.kind != "eof" else "end of file"
            raise MacroError(f"'{v}' expected but '{got}' found", tok.line)
        return tok

    def program(self) -> Program:
        prog = Program()
        while self.peek().kind != "eof":
            if self.is_kw("function"):
                f = self.function()
                prog.functions[f.name] = f
            elif self.is_kw("macro"):
                self.next()
                name_tok = self.next()
                if name_tok.kind != "str":
                    raise MacroError("Macro name expected", name_tok.line)
                prog.macros.append((name_tok.value, self.block()))
            else:
                prog.body.append(self.statement())
        return prog

    def function(self) -> FuncDef:
        line = self.next().line
        name = self.next()
        if name.kind != "id":
            raise MacroError("Function name expected", name.line)
        self.expect("(")
        params = []
        while not self.is_op(")"):
            tok = self.next()
            if tok.kind != "id":
                raise MacroError("Parameter name expected", tok.line)
            params.append(tok.value)
            if self.is_op(","):
                self.next()
        self.expect(")")
        return FuncDef(name.value, params, self.block(), line)

    def block(self) -> list:
        if not self.is_op("{"):
            return [self.statement()]
        self.next()
        out = []
        while not self.is_op("}"):
            if self.peek().kind == "eof":
                raise MacroError("'}' expected", self.peek().line)
            if self.is_kw("function"):
                raise MacroError("Nested functions are not supported", self.peek().line)
            out.append(self.statement())
        self.next()
        return out

    def statement(self):
        tok = self.peek()
        line = tok.line
        if tok.kind == "op" and tok.value == ";":
            self.next()
            return ("nop", line)
        if tok.kind == "op" and tok.value == "{":
            return ("block", line, self.block())
        if tok.kind == "id":
            kw = tok.value
            if kw == "var":
                self.next()
                decls = []
                while True:
                    name = self.next()
                    init = None
                    if self.is_op("="):
                        self.next()
                        init = self.expr()
                    decls.append((name.value, init))
                    if self.is_op(","):
                        self.next()
                        continue
                    break
                self.semi()
                return ("var", line, decls)
            if kw == "if":
                self.next()
                self.expect("(")
                cond = self.expr()
                self.expect(")")
                then = self.block()
                other = None
                if self.is_kw("else"):
                    self.next()
                    other = self.block()
                return ("if", line, cond, then, other)
            if kw == "for":
                self.next()
                self.expect("(")
                init = None if self.is_op(";") else self.expr()
                self.expect(";")
                cond = None if self.is_op(";") else self.expr()
                self.expect(";")
                step = None if self.is_op(")") else self.expr()
                self.expect(")")
                return ("for", line, init, cond, step, self.block())
            if kw == "while":
                self.next()
                self.expect("(")
                cond = self.expr()
                self.expect(")")
                return ("while", line, cond, self.block())
            if kw == "do":
                self.next()
                body = self.block()
                if not self.is_kw("while"):
                    raise MacroError("'while' expected", self.peek().line)
                self.next()
                self.expect("(")
                cond = self.expr()
                self.expect(")")
                self.semi()
                return ("do", line, body, cond)
            if kw == "return":
                self.next()
                val = None if self.is_op(";") or self.is_op("}") else self.expr()
                self.semi()
                return ("return", line, val)
            if kw in ("break", "continue"):
                self.next()
                self.semi()
                return (kw, line)
            if kw == "requires":
                self.next()
                self.expect("(")
                self.expr()
                self.expect(")")
                self.semi()
                return ("nop", line)
        e = self.expr()
        self.semi()
        return ("expr", line, e)

    def semi(self) -> None:
        if self.is_op(";"):
            self.next()

    # expressions --------------------------------------------------------------------
    def expr(self):
        return self.assignment()

    def assignment(self):
        left = self.ternary()
        tok = self.peek()
        if tok.kind == "op" and tok.value in ("=", "+=", "-=", "*=", "/="):
            if left[0] not in ("var", "index"):
                raise MacroError("Variable expected before '='", tok.line)
            self.next()
            right = self.assignment()
            return ("assign", tok.line, tok.value, left, right)
        return left

    def ternary(self):
        cond = self.logical()
        if self.is_op("?"):
            line = self.next().line
            a = self.assignment()
            self.expect(":")
            b = self.assignment()
            return ("ternary", line, cond, a, b)
        return cond

    def logical(self):
        left = self.comparison()
        while self.peek().kind == "op" and self.peek().value in ("&&", "||"):
            op = self.next()
            right = self.comparison()
            left = ("logic", op.line, op.value, left, right)
        return left

    def comparison(self):
        left = self.additive()
        while self.peek().kind == "op" and self.peek().value in ("==", "!=", "<", ">", "<=",
                                                                  ">="):
            op = self.next()
            right = self.additive()
            left = ("bin", op.line, op.value, left, right)
        return left

    def additive(self):
        left = self.term()
        while self.peek().kind == "op" and self.peek().value in ("+", "-"):
            op = self.next()
            right = self.term()
            left = ("bin", op.line, op.value, left, right)
        return left

    def term(self):
        left = self.unary()
        while self.peek().kind == "op" and self.peek().value in ("*", "/", "%", "&", "|", "^",
                                                                  "<<", ">>"):
            op = self.next()
            right = self.unary()
            left = ("bin", op.line, op.value, left, right)
        return left

    def unary(self):
        tok = self.peek()
        if tok.kind == "op" and tok.value in ("-", "!", "~", "+"):
            self.next()
            return ("unary", tok.line, tok.value, self.unary())
        if tok.kind == "op" and tok.value in ("++", "--"):
            self.next()
            target = self.unary()
            return ("incdec", tok.line, tok.value, target, True)
        return self.postfix()

    def postfix(self):
        node = self.primary()
        while True:
            tok = self.peek()
            if tok.kind != "op":
                break
            if tok.value == "[":
                self.next()
                idx = self.expr()
                self.expect("]")
                node = ("index", tok.line, node, idx)
            elif tok.value == "." and self.peek(1).kind == "id":
                self.next()
                name = self.next().value
                if self.is_op("("):
                    args = self.args()
                    node = ("method", tok.line, node, name, args)
                else:
                    node = ("prop", tok.line, node, name)
            elif tok.value in ("++", "--"):
                self.next()
                node = ("incdec", tok.line, tok.value, node, False)
            else:
                break
        return node

    def args(self) -> list:
        self.expect("(")
        out = []
        while not self.is_op(")"):
            out.append(self.expr())
            if self.is_op(","):
                self.next()
            elif not self.is_op(")"):
                raise MacroError("',' or ')' expected", self.peek().line)
        self.expect(")")
        return out

    def primary(self):
        tok = self.next()
        if tok.kind == "num":
            return ("num", tok.line, tok.value)
        if tok.kind == "str":
            return ("str", tok.line, tok.value)
        if tok.kind == "op" and tok.value == "(":
            e = self.expr()
            self.expect(")")
            return e
        if tok.kind == "id":
            name = tok.value
            if name in NAMESPACES and self.is_op(".") and self.peek(1).kind == "id":
                self.next()
                name = name + "." + self.next().value
            if self.is_op("("):
                return ("call", tok.line, name, self.args())
            return ("var", tok.line, name)
        got = tok.value if tok.kind != "eof" else "end of file"
        raise MacroError(f"Number or numeric function expected; found '{got}'", tok.line)


def parse(src: str) -> Program:
    return Parser(tokenize(src)).program()


# ---------------------------------------------------------------------------------- values
def n2s(x: float) -> str:
    """ImageJ's number -> string in string expressions (Interpreter.toString)."""
    x = float(x)
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if abs(x) < 2 ** 31 and x == int(x):
        return str(int(x))
    s = d2s_sig(x, 4, 9)
    if "E" not in s and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def d2s_sig(x: float, significant: int, max_digits: int) -> str:
    """``IJ.d2s(x, significantDigits, maxDigits)``."""
    from rsstatchem.core.imaging.io.text_image import d2s

    if x == 0:
        return d2s(x, 0) if significant <= 0 else d2s(0.0, 0)
    log10 = math.log10(abs(x))
    round_err = 0.223 * 10 ** (-max_digits)
    magnitude = int(math.ceil(log10 + round_err))
    decimals = max_digits - magnitude
    if decimals < 0 or magnitude < significant + 1 - max_digits:
        return d2s(x, -significant)
    if decimals > significant:
        decimals = max(significant, decimals - max_digits + significant)
    return d2s(x, decimals)


def to_str(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return ",".join(to_str(e) for e in v)
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return ""
    return n2s(v)


def to_num(v, line: int = 0) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            raise MacroError(f'Number expected: "{v}"', line) from None
    raise MacroError("Number expected", line)


def truthy(v) -> bool:
    if isinstance(v, str):
        return v.lower() == "true" or (v not in ("", "0", "false"))
    return bool(to_num(v))


# ---------------------------------------------------------------------------------- interpreter
@dataclass
class UIHooks:
    """Optional callbacks the ImageJ bar provides (all may be None when headless)."""
    get_number: Callable | None = None       # (prompt, default) -> float | None
    get_string: Callable | None = None       # (prompt, default) -> str | None
    get_boolean: Callable | None = None      # (prompt) -> bool | None
    show_message: Callable | None = None     # (title, text)
    dialog: Callable | None = None           # (title, fields) -> dict | None
    wait_for_user: Callable | None = None    # (title, text)
    update_display: Callable | None = None   # ()
    set_tool: Callable | None = None         # (name_or_id)
    open_path: Callable | None = None        # (path) -> object


class Interpreter:
    def __init__(self, session, ui: UIHooks | None = None, argument: str = "",
                 path: str | None = None):
        self.session = session
        self.ui = ui or UIHooks()
        self.argument = argument
        self.path = path
        self.globals: dict[str, Any] = {}
        self.frames: list[dict[str, Any]] = [{}]
        self.functions: dict[str, FuncDef] = {}
        self.batch_mode = False
        self.line = 0
        self.files: dict[int, Any] = {}
        self.dialog: dict | None = None
        self.depth = 0
        self.colors = {"foreground": None}
        self.started = time.perf_counter()

    # ---------------------------------------------------------------- variables
    def lookup(self, name: str, line: int):
        frame = self.frames[-1]
        if name in frame:
            return frame[name]
        if name in self.globals:
            return self.globals[name]
        const = CONSTANTS.get(name)
        if const is not None:
            return const(self) if callable(const) else const
        if name in BUILTINS:
            return self.call(name, [], line)
        raise MacroError(f'Undefined variable "{name}"', line)

    def assign(self, name: str, value) -> None:
        frame = self.frames[-1]
        if name in frame or name not in self.globals:
            frame[name] = value
        else:
            self.globals[name] = value

    def set_target(self, node, value) -> None:
        if node[0] == "var":
            self.assign(node[2], value)
        elif node[0] == "index":
            arr = self.eval(node[2])
            if not isinstance(arr, list):
                raise MacroError("Array expected", node[1])
            i = int(to_num(self.eval(node[3]), node[1]))
            if not 0 <= i < len(arr):
                raise MacroError(f"Index ({i}) out of 0-{len(arr) - 1} range", node[1])
            arr[i] = value
        else:
            raise MacroError("Variable expected", node[1])

    # ---------------------------------------------------------------- running
    def run_program(self, prog: Program, macro_name: str | None = None):
        self.functions.update(prog.functions)
        body = prog.body
        if macro_name is not None:
            match = [b for n, b in prog.macros if n == macro_name]
            if not match:
                raise MacroError(f'Macro "{macro_name}" not found')
            self.exec_block(prog.body)
            body = match[0]
        elif not [s for s in prog.body if s[0] not in ("var", "nop")] and prog.macros:
            self.exec_block(prog.body)
            body = prog.macros[0][1]
        try:
            self.exec_block(body)
        except _Return as r:
            return r.value
        except (_Break, _Continue):
            raise MacroError("'break' or 'continue' outside a loop", self.line) from None
        return None

    def exec_block(self, stmts) -> None:
        for s in stmts:
            self.exec(s)

    def exec(self, s) -> None:
        kind = s[0]
        self.line = s[1]
        if kind == "expr":
            self.eval(s[2])
        elif kind == "var":
            for name, init in s[2]:
                val = self.eval(init) if init is not None else 0.0
                if len(self.frames) == 1:
                    self.globals[name] = val
                else:
                    self.frames[-1][name] = val
        elif kind == "if":
            if truthy(self.eval(s[2])):
                self.exec_block(s[3])
            elif s[4] is not None:
                self.exec_block(s[4])
        elif kind == "for":
            if s[2] is not None:
                self.eval(s[2])
            while s[3] is None or truthy(self.eval(s[3])):
                try:
                    self.exec_block(s[5])
                except _Break:
                    break
                except _Continue:
                    pass
                if s[4] is not None:
                    self.eval(s[4])
                self._tick()
        elif kind == "while":
            while truthy(self.eval(s[2])):
                try:
                    self.exec_block(s[3])
                except _Break:
                    break
                except _Continue:
                    pass
                self._tick()
        elif kind == "do":
            while True:
                try:
                    self.exec_block(s[2])
                except _Break:
                    break
                except _Continue:
                    pass
                if not truthy(self.eval(s[3])):
                    break
                self._tick()
        elif kind == "block":
            self.exec_block(s[2])
        elif kind == "return":
            raise _Return(self.eval(s[2]) if s[2] is not None else None)
        elif kind == "break":
            raise _Break()
        elif kind == "continue":
            raise _Continue()

    def _tick(self) -> None:
        if getattr(self.session, "cancel_requested", False):
            self.session.cancel_requested = False
            raise MacroError("Macro canceled", self.line)

    # ---------------------------------------------------------------- expressions
    def eval(self, e):
        kind = e[0]
        if kind == "num":
            return e[2]
        if kind == "str":
            return e[2]
        if kind == "var":
            return self.lookup(e[2], e[1])
        if kind == "call":
            return self.call(e[2], e[3], e[1])
        if kind == "bin":
            return self.binop(e[2], self.eval(e[3]), self.eval(e[4]), e[1])
        if kind == "logic":
            a = truthy(self.eval(e[3]))
            if e[2] == "&&":
                return 1.0 if a and truthy(self.eval(e[4])) else 0.0
            return 1.0 if a or truthy(self.eval(e[4])) else 0.0
        if kind == "unary":
            v = self.eval(e[3])
            if e[2] == "-":
                return -to_num(v, e[1])
            if e[2] == "+":
                return to_num(v, e[1])
            if e[2] == "!":
                return 0.0 if truthy(v) else 1.0
            return float(~int(to_num(v, e[1])))
        if kind == "assign":
            op, target, rhs = e[2], e[3], self.eval(e[4])
            if op != "=":
                cur = self.eval(target)
                rhs = self.binop(op[0], cur, rhs, e[1])
            self.set_target(target, rhs)
            return rhs
        if kind == "incdec":
            target = e[3]
            old = to_num(self.eval(target), e[1])
            new = old + (1 if e[2] == "++" else -1)
            self.set_target(target, new)
            return new if e[4] else old
        if kind == "index":
            arr = self.eval(e[2])
            i = to_num(self.eval(e[3]), e[1])
            if isinstance(arr, str):
                return arr[int(i)]
            if not isinstance(arr, list):
                raise MacroError("Array expected", e[1])
            i = int(i)
            if not 0 <= i < len(arr):
                raise MacroError(f"Index ({i}) out of 0-{len(arr) - 1} range", e[1])
            return arr[i]
        if kind == "prop":
            obj = self.eval(e[2])
            if e[3] == "length" and isinstance(obj, (list, str)):
                return float(len(obj))
            raise MacroError(f'Unrecognized property "{e[3]}"', e[1])
        if kind == "method":
            obj = self.eval(e[2])
            args = [self.eval(a) for a in e[4]]
            return string_method(obj, e[3], args, e[1])
        if kind == "ternary":
            return self.eval(e[3]) if truthy(self.eval(e[2])) else self.eval(e[4])
        raise MacroError(f"Unsupported expression {kind}", e[1])

    def binop(self, op: str, a, b, line: int):
        if op == "+" and (isinstance(a, str) or isinstance(b, str)):
            return to_str(a) + to_str(b)
        if op in ("==", "!=", "<", ">", "<=", ">="):
            if isinstance(a, str) and isinstance(b, str):
                ca, cb = a.lower(), b.lower()
            elif isinstance(a, str) or isinstance(b, str):
                ca, cb = (to_str(a).lower(), to_str(b).lower()) if op in ("==", "!=") else \
                    (to_num(a, line), to_num(b, line))
            else:
                ca, cb = to_num(a, line), to_num(b, line)
            res = {"==": ca == cb, "!=": ca != cb, "<": ca < cb, ">": ca > cb, "<=": ca <= cb,
                   ">=": ca >= cb}[op]
            return 1.0 if res else 0.0
        x, y = to_num(a, line), to_num(b, line)
        if op == "+":
            return x + y
        if op == "-":
            return x - y
        if op == "*":
            return x * y
        if op == "/":
            if y == 0:
                return math.copysign(math.inf, x) if x != 0 else math.nan
            return x / y
        if op == "%":
            return math.fmod(x, y) if y != 0 else math.nan
        ix, iy = int(x), int(y)
        if op == "&":
            return float(ix & iy)
        if op == "|":
            return float(ix | iy)
        if op == "^":
            return float(ix ^ iy)
        if op == "<<":
            return float(ix << iy)
        if op == ">>":
            return float(ix >> iy)
        raise MacroError(f"Unknown operator {op}", line)

    def call(self, name: str, arg_nodes: list, line: int):
        self.line = line
        f = self.functions.get(name)
        if f is not None:
            if len(arg_nodes) != len(f.params):
                raise MacroError(f'"{name}" expects {len(f.params)} argument(s)', line)
            values = [self.eval(a) for a in arg_nodes]
            if self.depth > 500:
                raise MacroError("Stack overflow", line)
            self.frames.append(dict(zip(f.params, values)))
            self.depth += 1
            try:
                self.exec_block(f.body)
            except _Return as r:
                return r.value
            finally:
                self.frames.pop()
                self.depth -= 1
            return None
        spec = BUILTINS.get(name)
        if spec is None:
            raise MacroError(f'Unrecognized function: "{name}"', line)
        fn, refs = spec
        if refs:
            return fn(self, arg_nodes, line)
        return fn(self, [self.eval(a) for a in arg_nodes], line)

    # ---------------------------------------------------------------- image helpers
    def image(self, line: int = 0):
        im = self.session.active
        if im is None:
            raise MacroError("No image", line)
        return im

    def updated(self, im) -> None:
        if im in self.session.images:
            self.session.image_updated(im)


# ---------------------------------------------------------------------------------- string methods
def string_method(obj, name: str, args: list, line: int):
    if isinstance(obj, list):
        if name == "length":
            return float(len(obj))
        raise MacroError(f'Unrecognized array method "{name}"', line)
    if not isinstance(obj, str):
        obj = to_str(obj)
    a = [to_str(x) if not isinstance(x, (int, float)) else x for x in args]
    if name == "length":
        return float(len(obj))
    if name == "contains":
        return 1.0 if to_str(a[0]) in obj else 0.0
    if name == "startsWith":
        return 1.0 if obj.startswith(to_str(a[0])) else 0.0
    if name == "endsWith":
        return 1.0 if obj.endswith(to_str(a[0])) else 0.0
    if name == "indexOf":
        return float(obj.find(to_str(a[0])))
    if name == "lastIndexOf":
        return float(obj.rfind(to_str(a[0])))
    if name == "charAt":
        return obj[int(to_num(a[0]))]
    if name == "substring":
        i = int(to_num(a[0]))
        return obj[i:int(to_num(a[1]))] if len(a) > 1 else obj[i:]
    if name in ("toLowerCase", "toUpperCase"):
        return obj.lower() if name == "toLowerCase" else obj.upper()
    if name == "trim":
        return obj.strip()
    if name == "replace":
        return _java_replace(obj, to_str(a[0]), to_str(a[1]))
    if name == "matches":
        return 1.0 if re.fullmatch(_java_regex(to_str(a[0])), obj) else 0.0
    if name == "equals":
        return 1.0 if obj == to_str(a[0]) else 0.0
    raise MacroError(f'Unrecognized String function "{name}"', line)


def _java_regex(rx: str) -> str:
    return rx


def _java_replace(s: str, old: str, new: str) -> str:
    """ImageJ ``replace``: regular expression unless the pattern is a single special
    character or does not compile."""
    if len(old) == 1 or old in ("\\", "^", "$", ".", "|", "?", "*", "+", "(", ")", "[", "{"):
        return s.replace(old, new)
    try:
        repl = re.sub(r"\$(\d)", r"\\\1", new)
        return re.sub(old, repl, s)
    except re.error:
        return s.replace(old, new)


# ---------------------------------------------------------------------------------- builtins
BUILTINS: dict[str, tuple[Callable, bool]] = {}
CONSTANTS: dict[str, Any] = {"true": 1.0, "false": 0.0, "PI": math.pi, "NaN": math.nan}


def builtin(*names, refs: bool = False):
    def deco(fn):
        for n in names:
            BUILTINS[n] = (fn, refs)
        return fn
    return deco


def _n(args, i, default=None, line=0):
    if i < len(args):
        return to_num(args[i], line)
    if default is None:
        raise MacroError("Missing argument", line)
    return default


def _s(args, i, default=None):
    if i < len(args):
        return to_str(args[i])
    if default is None:
        raise MacroError("Missing argument")
    return default


def _out(it: Interpreter, nodes, values, line) -> None:
    """Assign output arguments (``getDimensions(w, h, ...)``)."""
    for node, v in zip(nodes, values):
        if node[0] not in ("var", "index"):
            raise MacroError("Variable expected", line)
        it.set_target(node, v)


# ---- output / misc
@builtin("print", "IJ.log")
def _print(it, args, line):
    if args and isinstance(args[0], dict) and "file" in args[0]:
        fh = args[0]
        fh["lines"].append(" ".join(to_str(a) for a in args[1:]))
        return None
    text = " ".join(to_str(a) for a in args)
    if text == "\\Clear":
        it.session.log_lines.clear()
        it.session.emit("log", "\\Clear")
        return None
    it.session.log(text)
    return None


@builtin("showStatus", "IJ.showStatus")
def _status(it, args, line):
    it.session.status(_s(args, 0, ""))


@builtin("showProgress", "IJ.showProgress")
def _progress(it, args, line):
    if len(args) == 2:
        it.session.emit("progress", (_n(args, 0) + 1) / max(_n(args, 1), 1))
    else:
        it.session.emit("progress", _n(args, 0, 1.0))


@builtin("wait")
def _wait(it, args, line):
    time.sleep(min(_n(args, 0, 0) / 1000.0, 5.0))


@builtin("exit")
def _exit(it, args, line):
    if args and it.ui.show_message is not None:
        it.ui.show_message("Macro", to_str(args[0]))
    elif args:
        it.session.log(to_str(args[0]))
    raise MacroExit(to_str(args[0]) if args else "")


@builtin("setBatchMode")
def _batch(it, args, line):
    v = args[0] if args else True
    it.batch_mode = truthy(v) if not isinstance(v, str) or v.lower() in ("true", "false") \
        else it.batch_mode
    return None


@builtin("is")
def _is(it, args, line):
    key = _s(args, 0).lower()
    im = it.session.active
    if key == "batch mode":
        return 1.0 if it.batch_mode else 0.0
    if im is None:
        return 0.0
    if key == "inverting lut":
        return 1.0 if im.is_inverted_lut() else 0.0
    if key == "hyperstack":
        return 1.0 if im.hyperstack or sum(d > 1 for d in im.dims) > 1 else 0.0
    if key == "composite":
        return 1.0 if im.is_composite else 0.0
    if key == "changes":
        return 1.0 if im.changed else 0.0
    if key == "binary":
        p = im.pixels
        return 1.0 if im.bit_depth == 8 and np.isin(p, (0, 255)).all() else 0.0
    if key == "grayscale":
        return 1.0 if not im.is_rgb and im.lut.is_grayscale() else 0.0
    if key in ("area", "line"):
        roi = im.roi
        return 1.0 if roi is not None and (roi.is_area() if key == "area" else roi.is_line()) \
            else 0.0
    if key == "global scale":
        return 0.0
    return 0.0


@builtin("getTime")
def _time(it, args, line):
    return float(int(time.time() * 1000))


@builtin("getArgument")
def _arg(it, args, line):
    return it.argument


@builtin("eval")
def _eval(it, args, line):
    code = _s(args, len(args) - 1)
    if len(args) > 1 and to_str(args[0]).lower() in ("python", "script"):
        return run_python(it.session, code)
    sub = Interpreter(it.session, it.ui)
    sub.run_program(parse(code))
    return None


@builtin("runMacro")
def _runmacro(it, args, line):
    path = Path(_s(args, 0))
    if not path.suffix:
        path = path.with_suffix(".ijm")
    return run_macro(it.session, path.read_text(), _s(args, 1, ""), it.ui, str(path))


@builtin("setOption")
def _setoption(it, args, line):
    name = _s(args, 0).lower()
    val = truthy(args[1]) if len(args) > 1 else True
    opts = it.session.options
    if name == "blackbackground":
        opts.black_background = val
    elif name in ("scaleconversions",):
        opts.scale_conversions = val
    else:
        opts.extra[f"option.{name}"] = val


# ---- math
_MATH1 = {"abs": abs, "sqrt": lambda x: math.sqrt(x) if x >= 0 else math.nan,
          "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "atan": math.atan, "asin": lambda x: math.asin(x) if -1 <= x <= 1 else math.nan,
          "acos": lambda x: math.acos(x) if -1 <= x <= 1 else math.nan,
          "floor": math.floor, "round": lambda x: math.floor(x + 0.5),
          "log": lambda x: math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan),
          "isNaN": lambda x: 1.0 if x != x else 0.0}
for _name, _fn in _MATH1.items():
    def _mk(fn):
        return lambda it, args, line: float(fn(_n(args, 0, line=line)))
    BUILTINS[_name] = (_mk(_fn), False)
    BUILTINS["Math." + _name] = (_mk(_fn), False)
BUILTINS["Math.ceil"] = (lambda it, a, ln: float(math.ceil(_n(a, 0))), False)
BUILTINS["Math.log10"] = (lambda it, a, ln: math.log10(_n(a, 0)) if _n(a, 0) > 0 else math.nan,
                          False)
BUILTINS["Math.signum"] = (lambda it, a, ln: float(np.sign(_n(a, 0))), False)
BUILTINS["Math.sqr"] = (lambda it, a, ln: _n(a, 0) ** 2, False)
BUILTINS["Math.constrain"] = (lambda it, a, ln: min(max(_n(a, 0), _n(a, 1)), _n(a, 2)), False)


@builtin("pow", "Math.pow")
def _pow(it, args, line):
    try:
        return math.pow(_n(args, 0), _n(args, 1))
    except (ValueError, OverflowError):
        return math.nan


@builtin("atan2", "Math.atan2")
def _atan2(it, args, line):
    return math.atan2(_n(args, 0), _n(args, 1))


@builtin("minOf", "Math.min")
def _min(it, args, line):
    return min(to_num(a) for a in args)


@builtin("maxOf", "Math.max")
def _max(it, args, line):
    return max(to_num(a) for a in args)


@builtin("random")
def _random(it, args, line):
    if args and to_str(args[0]).lower() == "seed":
        random.seed(int(_n(args, 1)))
        return None
    if args and to_str(args[0]).lower() == "gaussian":
        return random.gauss(0, 1)
    return random.random()


@builtin("parseInt")
def _parseint(it, args, line):
    s = _s(args, 0).strip()
    base = int(_n(args, 1, 10))
    try:
        return float(int(s, base))
    except ValueError:
        return math.nan


@builtin("parseFloat")
def _parsefloat(it, args, line):
    try:
        return float(_s(args, 0).strip().replace("Infinity", "inf"))
    except ValueError:
        return math.nan


@builtin("d2s")
def _d2s(it, args, line):
    from rsstatchem.core.imaging.io.text_image import d2s
    return d2s(_n(args, 0), int(_n(args, 1)))


@builtin("toString")
def _tostring(it, args, line):
    if len(args) > 1:
        return _d2s(it, args, line)
    return to_str(args[0])


@builtin("toHex")
def _tohex(it, args, line):
    return format(int(_n(args, 0)) & 0xFFFFFFFF, "x")


@builtin("toBinary")
def _tobin(it, args, line):
    return format(int(_n(args, 0)), "b")


@builtin("IJ.pad", "String.pad")
def _pad(it, args, line):
    s = args[0]
    s = str(int(s)) if isinstance(s, float) and s == int(s) else to_str(s)
    n = int(_n(args, 1))
    neg = s.startswith("-")
    body = s[1:] if neg else s
    return ("-" if neg else "") + body.rjust(n, "0")


# ---- strings
@builtin("lengthOf")
def _len(it, args, line):
    a = args[0]
    return float(len(a)) if isinstance(a, (str, list)) else float(len(to_str(a)))


@builtin("substring")
def _substring(it, args, line):
    s = _s(args, 0)
    i = int(_n(args, 1))
    return s[i:int(_n(args, 2))] if len(args) > 2 else s[i:]


@builtin("indexOf")
def _indexof(it, args, line):
    s, sub = _s(args, 0), _s(args, 1)
    return float(s.find(sub, int(_n(args, 2, 0))))


@builtin("lastIndexOf")
def _lastindexof(it, args, line):
    return float(_s(args, 0).rfind(_s(args, 1)))


@builtin("startsWith")
def _startswith(it, args, line):
    return 1.0 if _s(args, 0).startswith(_s(args, 1)) else 0.0


@builtin("endsWith")
def _endswith(it, args, line):
    return 1.0 if _s(args, 0).endswith(_s(args, 1)) else 0.0


@builtin("matches")
def _matches(it, args, line):
    return 1.0 if re.fullmatch(_s(args, 1), _s(args, 0)) else 0.0


@builtin("replace")
def _replace(it, args, line):
    return _java_replace(_s(args, 0), _s(args, 1), _s(args, 2))


@builtin("toLowerCase")
def _lower(it, args, line):
    return _s(args, 0).lower()


@builtin("toUpperCase")
def _upper(it, args, line):
    return _s(args, 0).upper()


@builtin("String.trim", "trim")
def _trim(it, args, line):
    return _s(args, 0).strip()


@builtin("split")
def _split(it, args, line):
    s = _s(args, 0)
    if len(args) < 2:
        return [t for t in re.split(r"[ \t\n\r]+", s) if t]
    delim = _s(args, 1)
    if delim == "\n":
        return [p for p in s.split("\n") if p != ""]
    if delim.startswith("(") and delim.endswith(")"):
        return re.split(delim[1:-1], s)
    if delim == ",":
        return s.split(",")
    return [t for t in re.split("[" + re.escape(delim) + "]+", s) if t]


@builtin("String.join")
def _join(it, args, line):
    arr = args[0] if isinstance(args[0], list) else [args[0]]
    return _s(args, 1, ",").join(to_str(a) for a in arr)


@builtin("fromCharCode")
def _fromchar(it, args, line):
    return "".join(chr(int(to_num(a))) for a in args)


@builtin("charCodeAt")
def _charcode(it, args, line):
    return float(ord(_s(args, 0)[int(_n(args, 1))]))


@builtin("String.format")
def _format(it, args, line):
    fmt = _s(args, 0)
    try:
        return fmt % tuple(to_num(a) for a in args[1:])
    except (TypeError, ValueError):
        return fmt


# ---- arrays
@builtin("newArray")
def _newarray(it, args, line):
    if len(args) == 1 and not isinstance(args[0], str):
        return [0.0] * int(to_num(args[0]))
    return list(args)


@builtin("Array.concat")
def _aconcat(it, args, line):
    out = []
    for a in args:
        out.extend(a if isinstance(a, list) else [a])
    return out


@builtin("Array.copy")
def _acopy(it, args, line):
    return list(args[0])


@builtin("Array.slice")
def _aslice(it, args, line):
    a = args[0]
    i = int(_n(args, 1))
    return a[i:int(_n(args, 2))] if len(args) > 2 else a[i:]


@builtin("Array.reverse", "Array.invert")
def _areverse(it, args, line):
    args[0].reverse()
    return args[0]


@builtin("Array.sort")
def _asort(it, args, line):
    a = args[0]
    if all(isinstance(v, str) for v in a):
        a.sort(key=str.lower)
    else:
        a.sort(key=lambda v: to_num(v))
    return a


@builtin("Array.getSequence")
def _aseq(it, args, line):
    return [float(i) for i in range(int(_n(args, 0)))]


@builtin("Array.fill")
def _afill(it, args, line):
    a = args[0]
    for i in range(len(a)):
        a[i] = args[1]
    return a


@builtin("Array.getStatistics", refs=True)
def _astats(it, nodes, line):
    a = [to_num(v) for v in it.eval(nodes[0])]
    arr = np.array(a, float)
    vals = [float(arr.min()), float(arr.max()), float(arr.mean()),
            float(arr.std(ddof=1)) if len(arr) > 1 else 0.0]
    _out(it, nodes[1:], vals, line)


@builtin("Array.print")
def _aprint(it, args, line):
    it.session.log(",".join(to_str(v) for v in args[0]))


@builtin("Array.findMaxima", "Array.findMinima")
def _amax(it, args, line):
    a = np.array([to_num(v) for v in args[0]])
    order = np.argsort(-a, kind="stable")
    return [float(i) for i in order]


@builtin("Array.rankPositions")
def _arank(it, args, line):
    a = args[0]
    keys = [to_num(v) if not isinstance(v, str) else v for v in a]
    return [float(i) for i in sorted(range(len(a)), key=lambda i: keys[i])]


@builtin("Array.deleteValue")
def _adelete(it, args, line):
    return [v for v in args[0] if not (to_str(v) == to_str(args[1]))]


@builtin("Array.trim")
def _atrim(it, args, line):
    return args[0][:int(_n(args, 1))]


@builtin("Array.resample")
def _aresample(it, args, line):
    a = np.array([to_num(v) for v in args[0]])
    n = int(_n(args, 1))
    return [float(v) for v in np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)]


# ---- dialogs (UI hooks or defaults)
@builtin("getNumber")
def _getnumber(it, args, line):
    default = _n(args, 1, 0)
    if it.ui.get_number is not None:
        v = it.ui.get_number(_s(args, 0), default)
        if v is None:
            raise MacroExit("")
        return float(v)
    return default


@builtin("getString")
def _getstring(it, args, line):
    default = _s(args, 1, "")
    if it.ui.get_string is not None:
        v = it.ui.get_string(_s(args, 0), default)
        if v is None:
            raise MacroExit("")
        return v
    return default


@builtin("getBoolean")
def _getboolean(it, args, line):
    if it.ui.get_boolean is not None:
        v = it.ui.get_boolean(_s(args, 0))
        if v is None:
            raise MacroExit("")
        return 1.0 if v else 0.0
    return 1.0


@builtin("showMessage", "waitForUser", "showMessageWithCancel")
def _showmessage(it, args, line):
    title, text = ("Message", _s(args, 0, "")) if len(args) < 2 else (_s(args, 0), _s(args, 1))
    if it.ui.show_message is not None:
        it.ui.show_message(title, text)
    else:
        it.session.log(text)


@builtin("Dialog.create", "Dialog.createNonBlocking")
def _dcreate(it, args, line):
    it.dialog = {"title": _s(args, 0, "Dialog"), "fields": [], "values": [], "pos": 0}


def _dadd(kind):
    def fn(it, args, line):
        if it.dialog is None:
            raise MacroError("Dialog.create() must be called first", line)
        it.dialog["fields"].append((kind, args))
    return fn


for _k in ("Number", "String", "Checkbox", "Choice", "Slider", "Message", "Radio",
           "RadioButtonGroup", "Directory", "File", "Help", "CheckboxGroup", "Image",
           "ImageChoice"):
    BUILTINS["Dialog.add" + _k] = (_dadd(_k), False)


@builtin("Dialog.show")
def _dshow(it, args, line):
    d = it.dialog
    if d is None:
        raise MacroError("Dialog.create() must be called first", line)
    values = []
    for kind, a in d["fields"]:
        if kind in ("Number", "Slider"):
            values.append(("n", to_num(a[-1] if kind == "Slider" else a[1])))
        elif kind in ("String", "Directory", "File"):
            values.append(("s", to_str(a[1]) if len(a) > 1 else ""))
        elif kind == "Checkbox":
            values.append(("b", truthy(a[1])))
        elif kind == "CheckboxGroup":
            for v in a[3]:
                values.append(("b", truthy(v)))
        elif kind in ("Choice", "RadioButtonGroup", "Radio"):
            items = a[1] if len(a) > 1 and isinstance(a[1], list) else a[3] if len(a) > 3 \
                else []
            default = to_str(a[2]) if kind == "Choice" and len(a) > 2 else \
                to_str(a[-1]) if a else ""
            values.append(("s", default if default else (to_str(items[0]) if items else "")))
        elif kind in ("Image", "ImageChoice"):
            values.append(("s", it.session.active.title if it.session.active else ""))
    if it.ui.dialog is not None:
        res = it.ui.dialog(d["title"], d["fields"])
        if res is None:
            raise MacroExit("")
        values = res
    d["values"] = values
    d["pos"] = {"n": 0, "s": 0, "b": 0}


def _dget(kind):
    def fn(it, args, line):
        d = it.dialog
        if d is None or not d.get("values"):
            raise MacroError("Dialog.show() must be called first", line)
        seq = [v for k, v in d["values"] if k == kind]
        i = d["pos"][kind]
        d["pos"][kind] += 1
        if i >= len(seq):
            raise MacroError("No more dialog values", line)
        v = seq[i]
        return (1.0 if v else 0.0) if kind == "b" else v
    return fn


BUILTINS["Dialog.getNumber"] = (_dget("n"), False)
BUILTINS["Dialog.getString"] = (_dget("s"), False)
BUILTINS["Dialog.getChoice"] = (_dget("s"), False)
BUILTINS["Dialog.getRadioButton"] = (_dget("s"), False)
BUILTINS["Dialog.getCheckbox"] = (_dget("b"), False)
BUILTINS["Dialog.getImageChoice"] = (_dget("s"), False)


# ---- images and windows
@builtin("run")
def _run(it, args, line):
    from rsstatchem.core.imaging.commands import CommandError

    label = _s(args, 0)
    options = _s(args, 1, "")
    fallback = FALLBACK_COMMANDS.get(label)
    if label not in it.session.registry and fallback is not None:
        fallback(it, options)
        return None
    try:
        it.session.run(label, options)
    except CommandError as exc:
        raise MacroError(str(exc), line) from None
    return None


def _select_none(it, options):
    im = it.session.active
    if im is not None:
        im.roi = None
        it.updated(im)


def _select_all(it, options):
    from rsstatchem.core.imaging.roi import RectRoi
    im = it.image()
    im.roi = RectRoi(0, 0, im.width, im.height)
    it.updated(im)


def _clear_results(it, options):
    it.session.results.reset()
    it.session.show_table(it.session.results)


def _close_cmd(it, options):
    im = it.session.active
    if im is not None:
        it.session.close_image(im)


FALLBACK_COMMANDS: dict[str, Callable] = {
    "Select None": _select_none, "Select All": _select_all, "Clear Results": _clear_results,
    "Close": _close_cmd,
}


@builtin("open")
def _open(it, args, line):
    from rsstatchem.core.imaging.io import open_in_session

    if not args:
        raise MacroError("open() without a path is not supported in batch macros", line)
    path = _s(args, 0)
    if it.ui.open_path is not None:
        it.ui.open_path(path)
    else:
        open_in_session(it.session, path)
    return None


@builtin("close")
def _close(it, args, line):
    s = it.session
    if not args:
        if s.active is not None:
            s.close_image(s.active)
        return None
    pattern = _s(args, 0)
    if pattern in ("*", "\\Others"):
        for im in list(s.images):
            if pattern == "*" or im is not s.active:
                s.close_image(im)
        return None
    if pattern in ("Results", "Log", "ROI Manager"):
        if pattern == "Results":
            s.results.reset()
        elif pattern == "ROI Manager":
            s.roi_manager.clear()
        s.emit("window_close_request", pattern)
        return None
    rx = re.compile("^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$")
    for im in list(s.images):
        if rx.match(im.title):
            s.close_image(im)
    return None


@builtin("selectWindow", "selectImage")
def _select(it, args, line):
    key = args[0]
    s = it.session
    if isinstance(key, str):
        im = next((i for i in s.images if i.title == key), None)
        if im is None:
            if key in s.tables or key in ("Log", "Results", "ROI Manager"):
                return None
            raise MacroError(f'"{key}" not found', line)
    else:
        k = int(to_num(key))
        im = s.find_image(k)
        if im is None:
            raise MacroError(f"Image {k} not found or no images are open", line)
    s.set_active(im)
    return None


@builtin("isOpen")
def _isopen(it, args, line):
    key = args[0]
    s = it.session
    if isinstance(key, str):
        return 1.0 if any(i.title == key for i in s.images) or key in s.tables else 0.0
    return 1.0 if s.find_image(int(to_num(key))) is not None else 0.0


@builtin("isActive")
def _isactive(it, args, line):
    im = it.session.active
    return 1.0 if im is not None and im.id == int(_n(args, 0)) else 0.0


@builtin("nImages")
def _nimages(it, args, line):
    return float(len(it.session.images))


@builtin("getTitle")
def _gettitle(it, args, line):
    return it.image(line).title


@builtin("getImageID")
def _getid(it, args, line):
    return float(it.image(line).id)


@builtin("getList")
def _getlist(it, args, line):
    kind = _s(args, 0).lower()
    if kind == "image.titles":
        return [im.title for im in it.session.images]
    if kind == "window.titles":
        return [t for t in it.session.tables]
    if kind == "luts":
        from rsstatchem.core.imaging.lut import NAMED_LUTS
        return list(NAMED_LUTS)
    return []


@builtin("rename")
def _rename(it, args, line):
    im = it.image(line)
    im.title = it.session.unique_title(_s(args, 0), exclude=im)
    it.updated(im)


@builtin("getWidth")
def _getwidth(it, args, line):
    return float(it.image(line).width)


@builtin("getHeight")
def _getheight(it, args, line):
    return float(it.image(line).height)


@builtin("bitDepth")
def _bitdepth(it, args, line):
    im = it.session.active
    return float(im.bit_depth) if im is not None else 0.0


@builtin("nSlices")
def _nslices(it, args, line):
    return float(it.image(line).stack_size)


@builtin("getDimensions", refs=True)
def _getdims(it, nodes, line):
    im = it.image(line)
    c, z, t = im.dims
    _out(it, nodes, [float(im.width), float(im.height), float(c), float(z), float(t)], line)


@builtin("getSliceNumber")
def _getslice(it, args, line):
    return float(it.image(line).current_index + 1)


@builtin("setSlice")
def _setslice(it, args, line):
    im = it.image(line)
    n = int(_n(args, 0))
    if not 1 <= n <= im.stack_size:
        raise MacroError(f'Argument must be >=1 and <={im.stack_size}', line)
    im.set_position(*im.position_of(n - 1))
    it.updated(im)


def _stack_setter(axis):
    def fn(it, args, line):
        im = it.image(line)
        c, z, t = im.position
        v = int(_n(args, 0)) - 1
        if axis == "c":
            c = v
        elif axis == "z":
            z = v
        else:
            t = v
        im.set_position(c, z, t)
        it.updated(im)
    return fn


BUILTINS["Stack.setSlice"] = (_stack_setter("z"), False)
BUILTINS["Stack.setChannel"] = (_stack_setter("c"), False)
BUILTINS["Stack.setFrame"] = (_stack_setter("t"), False)


@builtin("Stack.setPosition")
def _stack_setpos(it, args, line):
    im = it.image(line)
    im.set_position(int(_n(args, 0)) - 1, int(_n(args, 1)) - 1, int(_n(args, 2)) - 1)
    it.updated(im)


@builtin("Stack.getPosition", refs=True)
def _stack_getpos(it, nodes, line):
    c, z, t = it.image(line).position
    _out(it, nodes, [float(c + 1), float(z + 1), float(t + 1)], line)


@builtin("Stack.getDimensions", refs=True)
def _stack_getdims(it, nodes, line):
    _getdims(it, nodes, line)


@builtin("Stack.isHyperstack", "Stack.isHyperStack")
def _stack_ishs(it, args, line):
    im = it.image(line)
    return 1.0 if im.hyperstack or sum(d > 1 for d in im.dims) > 1 else 0.0


@builtin("Stack.setDimensions")
def _stack_setdims(it, args, line):
    im = it.image(line)
    c, z, t = int(_n(args, 0)), int(_n(args, 1)), int(_n(args, 2))
    if c * z * t != im.stack_size:
        raise MacroError("channels*slices*frames != stack size", line)
    planes = im.pixels.reshape((-1,) + im.pixels.shape[3:])
    im.replace_pixels(planes.reshape((t, z, c) + planes.shape[1:]), im.type)
    im.hyperstack = True
    it.updated(im)


@builtin("Stack.setDisplayMode")
def _stack_mode(it, args, line):
    im = it.image(line)
    m = _s(args, 0).lower()
    if m in ("composite", "color", "grayscale") and im.is_composite:
        im.composite_mode = m
        it.updated(im)


@builtin("Stack.getDisplayMode", refs=True)
def _stack_getmode(it, nodes, line):
    _out(it, nodes, [it.image(line).composite_mode], line)


@builtin("Stack.setActiveChannels")
def _stack_active(it, args, line):
    im = it.image(line)
    bits = _s(args, 0)
    im.active_channels = [ch == "1" for ch in bits.ljust(im.n_channels, "0")[:im.n_channels]]
    it.updated(im)


@builtin("Stack.getFrameInterval")
def _stack_finterval(it, args, line):
    return float(it.image(line).calibration.frame_interval)


@builtin("Stack.setFrameInterval")
def _stack_setfinterval(it, args, line):
    it.image(line).calibration.frame_interval = _n(args, 0)


@builtin("Stack.getStatistics", refs=True)
def _stack_stats(it, nodes, line):
    im = it.image(line)
    p = im.pixels.astype(np.float64)
    vals = [float(p.size if not im.is_rgb else p.size // 3), float(p.mean()), float(p.min()),
            float(p.max()), float(p.std(ddof=1)) if p.size > 1 else 0.0]
    _out(it, nodes, vals, line)


@builtin("newImage")
def _newimage(it, args, line):
    from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc

    title = _s(args, 0)
    spec = _s(args, 1).lower()
    w, h = int(_n(args, 2)), int(_n(args, 3))
    if len(args) >= 7:
        c, z, t = int(_n(args, 4)), int(_n(args, 5)), int(_n(args, 6))
    else:
        c, z, t = 1, int(_n(args, 4, 1)), 1
    typ = GRAY8
    if "16" in spec:
        typ = GRAY16
    elif "32" in spec:
        typ = GRAY32
    elif "rgb" in spec:
        typ = COLOR_RGB
    fill = "white" if "white" in spec else "ramp" if "ramp" in spec else \
        "random" if "random" in spec or "noise" in spec else "black"
    doc = ImageDoc.new(w, h, typ, c, z, t, title=title)
    _fill_new(doc, fill)
    if c > 1 or t > 1:
        doc.hyperstack = True
    it.session.add_image(doc)


def _fill_new(doc, fill: str) -> None:
    """NewImage fills: white, black, ramp (horizontal 0..max), random (gaussian noise)."""
    from rsstatchem.core.imaging.image import GRAY16, GRAY32

    w = doc.width
    if fill == "white":
        if doc.is_rgb or doc.bit_depth == 8:
            doc.pixels[...] = 255
        elif doc.type == GRAY16:
            doc.pixels[...] = 65535
        else:
            doc.pixels[...] = 1.0
    elif fill == "ramp":
        ramp = np.arange(w, dtype=np.float64) / w
        if doc.type == GRAY32:
            row = ramp.astype(np.float32)
        elif doc.type == GRAY16:
            row = (ramp * 65536).astype(np.uint16)
        else:
            row = (ramp * 256).astype(np.uint8)
        if doc.is_rgb:
            doc.pixels[...] = row[None, :, None]
        else:
            doc.pixels[...] = row[None, :]
    elif fill == "random":
        rng = np.random.default_rng()
        if doc.type == GRAY32:
            doc.pixels[...] = rng.normal(0, 1, doc.pixels.shape).astype(np.float32)
        elif doc.type == GRAY16:
            doc.pixels[...] = np.clip(rng.normal(32768, 4000, doc.pixels.shape), 0, 65535)
        else:
            doc.pixels[...] = np.clip(rng.normal(128, 40, doc.pixels.shape), 0, 255)
    doc.reset_display_range()


@builtin("saveAs")
def _saveas(it, args, line):
    from rsstatchem.core.imaging.io import save_in_session
    fmt = _s(args, 0)
    path = _s(args, 1, "")
    if not path:
        raise MacroError("saveAs() requires a path in macros", line)
    save_in_session(it.session, fmt, path)


@builtin("save")
def _save(it, args, line):
    from rsstatchem.core.imaging.io import save_in_session
    path = _s(args, 0)
    save_in_session(it.session, Path(path).suffix.lstrip(".") or "tiff", path)


@builtin("getPixel")
def _getpixel(it, args, line):
    im = it.image(line)
    x, y = _n(args, 0), _n(args, 1)
    p = im.plane()
    if x != int(x) or y != int(y):
        if im.is_rgb:
            xi, yi = int(x), int(y)
            r, g, b = (int(v) for v in p[yi, xi])
            return float((r << 16) | (g << 8) | b)
        return float(_bilinear(p, x, y))
    xi, yi = int(x), int(y)
    if not (0 <= xi < im.width and 0 <= yi < im.height):
        return 0.0
    if im.is_rgb:
        r, g, b = (int(v) for v in p[yi, xi])
        return float((r << 16) | (g << 8) | b)
    return float(p[yi, xi])


def _bilinear(p, x, y):
    h, w = p.shape[:2]
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    v = (p[y0, x0] * (1 - fx) + p[y0, x1] * fx) * (1 - fy) + \
        (p[y1, x0] * (1 - fx) + p[y1, x1] * fx) * fy
    return float(v)


@builtin("setPixel")
def _setpixel(it, args, line):
    im = it.image(line)
    x, y, v = int(_n(args, 0)), int(_n(args, 1)), _n(args, 2)
    if not (0 <= x < im.width and 0 <= y < im.height):
        return None
    p = im.plane()
    if im.is_rgb:
        iv = int(v)
        p[y, x] = ((iv >> 16) & 255, (iv >> 8) & 255, iv & 255)
    elif im.bit_depth == 8:
        p[y, x] = int(v) & 255
    elif im.bit_depth == 16:
        p[y, x] = int(v) & 0xFFFF
    else:
        p[y, x] = v
    im.changed = True
    return None


@builtin("updateDisplay")
def _updatedisplay(it, args, line):
    im = it.session.active
    if im is not None:
        it.updated(im)


@builtin("getMinAndMax", refs=True)
def _getminmax(it, nodes, line):
    im = it.image(line)
    lo, hi = im.display_range
    cal = im.calibration
    if cal.calibrated() and not im.is_rgb:
        lo, hi = cal.cvalue(lo), cal.cvalue(hi)
    _out(it, nodes, [float(lo), float(hi)], line)


@builtin("setMinAndMax")
def _setminmax(it, args, line):
    im = it.image(line)
    lo, hi = _n(args, 0), _n(args, 1)
    cal = im.calibration
    if cal.calibrated() and im.bit_depth in (8, 16):
        lo, hi = cal.raw_value(lo, im.bit_depth), cal.raw_value(hi, im.bit_depth)
    if im.is_rgb:
        im.display_ranges = [(lo, hi)]
    else:
        im.set_display_range(lo, hi)
    it.updated(im)


@builtin("resetMinAndMax")
def _resetminmax(it, args, line):
    im = it.image(line)
    im.reset_display_range(None if im.is_rgb else im.position[0])
    it.updated(im)


@builtin("getVoxelSize", refs=True)
def _getvoxel(it, nodes, line):
    cal = it.image(line).calibration
    _out(it, nodes, [cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.units], line)


@builtin("getPixelSize", refs=True)
def _getpixelsize(it, nodes, line):
    cal = it.image(line).calibration
    _out(it, nodes, [cal.units, cal.pixel_width, cal.pixel_height, cal.pixel_depth], line)


@builtin("setVoxelSize")
def _setvoxel(it, args, line):
    im = it.image(line)
    cal = im.calibration
    cal.pixel_width, cal.pixel_height, cal.pixel_depth = _n(args, 0), _n(args, 1), _n(args, 2)
    unit = _s(args, 3, "pixel")
    cal.unit = "µm" if unit in ("um", "micron", "microns") else unit
    it.updated(im)


@builtin("calibrate")
def _calibrate(it, args, line):
    return float(it.image(line).calibration.cvalue(_n(args, 0)))


@builtin("getMetadata")
def _getmetadata(it, args, line):
    im = it.image(line)
    key = _s(args, 0, "Label").lower()
    if key == "info":
        return im.info or ""
    label = im.slice_label() or ""
    if im.stack_size == 1 and not label:
        return im.info or ""
    return label


@builtin("setMetadata")
def _setmetadata(it, args, line):
    im = it.image(line)
    if len(args) == 1:
        key, value = "label", _s(args, 0)
    else:
        key, value = _s(args, 0).lower(), _s(args, 1)
    if key == "info":
        im.info = value
    else:
        im.set_slice_label(value)
    it.updated(im)


@builtin("Property.get")
def _propget(it, args, line):
    im = it.image(line)
    return im.properties.get(_s(args, 0), "")


@builtin("Property.set")
def _propset(it, args, line):
    im = it.image(line)
    key, value = _s(args, 0), _s(args, 1)
    if value == "":
        im.properties.pop(key, None)
    else:
        im.properties[key] = value


@builtin("getInfo")
def _getinfo(it, args, line):
    key = _s(args, 0, "")
    im = it.session.active
    low = key.lower()
    if low == "image.subtitle" and im is not None:
        return im.subtitle()
    if low == "image.title" and im is not None:
        return im.title
    if low in ("image.directory", "image.filename") and im is not None and im.path:
        return str(Path(im.path).parent) + "/" if low == "image.directory" else \
            Path(im.path).name
    if low == "selection.name" and im is not None and im.roi is not None:
        return im.roi.name
    if low == "macro.filepath":
        return it.path or "null"
    if low == "log":
        return "\n".join(it.session.log_lines)
    if low in ("os.name",):
        import platform
        return platform.system()
    if low == "micrometer.abbreviation":
        return "µm"
    if re.fullmatch(r"[0-9A-Fa-f]{4},[0-9A-Fa-f]{4}", key) and im is not None:
        for ln in (im.info or "").splitlines():
            if ln.upper().startswith(key.upper()):
                return ln.split(":", 1)[1].strip() if ":" in ln else ""
        return ""
    if im is not None and key in im.properties:
        return im.properties[key]
    if not key and im is not None:
        return im.info or ""
    return ""


# ---- selections
def _set_roi(it, roi, line):
    im = it.image(line)
    im.roi = roi
    it.updated(im)


@builtin("makeRectangle")
def _makerect(it, args, line):
    from rsstatchem.core.imaging.roi import RectRoi
    x, y, w, h = (_n(args, i) for i in range(4))
    _set_roi(it, RectRoi(x, y, w, h, corner_diameter=_n(args, 4, 0)) if w > 0 and h > 0
             else None, line)


@builtin("makeOval")
def _makeoval(it, args, line):
    from rsstatchem.core.imaging.roi import OvalRoi
    x, y, w, h = (_n(args, i) for i in range(4))
    _set_roi(it, OvalRoi(x, y, w, h) if w > 0 and h > 0 else None, line)


@builtin("makeEllipse")
def _makeellipse(it, args, line):
    from rsstatchem.core.imaging.roi import EllipseRoi
    _set_roi(it, EllipseRoi(*(_n(args, i) for i in range(4)), aspect_ratio=_n(args, 4)), line)


@builtin("makeRotatedRectangle")
def _makerotrect(it, args, line):
    from rsstatchem.core.imaging.roi import RotatedRectRoi
    _set_roi(it, RotatedRectRoi(*(_n(args, i) for i in range(4)), rect_width=_n(args, 4)),
             line)


@builtin("makePolygon")
def _makepolygon(it, args, line):
    from rsstatchem.core.imaging.roi import PolygonRoi
    vals = [to_num(a) for a in args]
    if len(vals) < 6:
        raise MacroError("At least 3 points required", line)
    _set_roi(it, PolygonRoi(vals[0::2], vals[1::2]), line)


@builtin("makeLine")
def _makeline(it, args, line):
    from rsstatchem.core.imaging.roi import LineRoi, PolygonRoi
    from rsstatchem.core.imaging.roi import shapes as S
    vals = [to_num(a) for a in args]
    if len(vals) == 4:
        lw = it.session.options.line_width
        _set_roi(it, LineRoi(*vals, width=lw, stroke_width=lw if lw > 1 else 0), line)
    else:
        _set_roi(it, PolygonRoi(vals[0::2], vals[1::2], type=S.POLYLINE), line)


@builtin("makeArrow")
def _makearrow(it, args, line):
    from rsstatchem.core.imaging.roi import LineRoi
    vals = [to_num(a) for a in args[:4]]
    _set_roi(it, LineRoi(*vals, arrow=True, stroke_width=2), line)


@builtin("makePoint")
def _makepoint(it, args, line):
    from rsstatchem.core.imaging.roi import PointRoi
    _set_roi(it, PointRoi([_n(args, 0)], [_n(args, 1)]), line)


@builtin("makeSelection")
def _makeselection(it, args, line):
    from rsstatchem.core.imaging.roi import PointRoi, PolygonRoi
    from rsstatchem.core.imaging.roi import shapes as S
    kind = args[0]
    if not isinstance(kind, str):
        kind = {2: "polygon", 3: "freehand", 4: "traced", 6: "polyline", 7: "freeline",
                8: "angle", 10: "point"}.get(int(to_num(kind)), "polygon")
    kind = kind.lower()
    if len(args) == 3 and isinstance(args[1], list):
        xs, ys = [to_num(v) for v in args[1]], [to_num(v) for v in args[2]]
    else:
        vals = [to_num(a) for a in args[1:]]
        xs, ys = vals[0::2], vals[1::2]
    if "point" in kind:
        _set_roi(it, PointRoi(xs, ys), line)
        return
    t = S.POLYGON
    for key, typ in (("freehand", S.FREEROI), ("traced", S.TRACED_ROI),
                     ("polyline", S.POLYLINE), ("freeline", S.FREELINE), ("angle", S.ANGLE),
                     ("line", S.POLYLINE)):
        if key in kind:
            t = typ
            break
    _set_roi(it, PolygonRoi(xs, ys, type=t), line)


@builtin("makeText")
def _maketext(it, args, line):
    from rsstatchem.core.imaging.roi import TextRoi
    _set_roi(it, TextRoi(x=_n(args, 1), y=_n(args, 2), text=_s(args, 0)), line)


@builtin("selectionType")
def _seltype(it, args, line):
    im = it.session.active
    if im is None or im.roi is None:
        return -1.0
    return float(im.roi.type)


@builtin("getSelectionBounds", refs=True)
def _selbounds(it, nodes, line):
    im = it.image(line)
    if im.roi is None:
        vals = [0.0, 0.0, float(im.width), float(im.height)]
    else:
        x, y, w, h = im.roi.bounds()
        x0, y0 = math.floor(x), math.floor(y)
        vals = [float(x0), float(y0), float(math.ceil(x + w) - x0) if not
                float(w).is_integer() else float(w),
                float(math.ceil(y + h) - y0) if not float(h).is_integer() else float(h)]
    _out(it, nodes, vals, line)


@builtin("getSelectionCoordinates", refs=True)
def _selcoords(it, nodes, line):
    im = it.image(line)
    if im.roi is None:
        raise MacroError("Selection required", line)
    xs, ys = im.roi.polygon()
    _out(it, nodes, [[float(v) for v in xs], [float(v) for v in ys]], line)


@builtin("getLine", refs=True)
def _getline(it, nodes, line):
    from rsstatchem.core.imaging.roi import LineRoi
    im = it.image(line)
    r = im.roi
    if isinstance(r, LineRoi):
        vals = [r.x1, r.y1, r.x2, r.y2, r.width]
    else:
        vals = [-1.0, -1.0, -1.0, -1.0, it.session.options.line_width]
    _out(it, nodes, [float(v) for v in vals], line)


@builtin("Roi.setName")
def _roiname(it, args, line):
    im = it.image(line)
    if im.roi is None:
        raise MacroError("Selection required", line)
    im.roi.name = _s(args, 0)


@builtin("Roi.getName")
def _roigetname(it, args, line):
    im = it.image(line)
    return im.roi.name if im.roi is not None else ""


@builtin("Roi.setStrokeColor")
def _roistroke(it, args, line):
    im = it.image(line)
    if im.roi is not None:
        im.roi.stroke_color = _color_arg(args)
        it.updated(im)


@builtin("Roi.setFillColor")
def _roifill(it, args, line):
    im = it.image(line)
    if im.roi is not None:
        im.roi.fill_color = _color_arg(args)
        it.updated(im)


@builtin("Roi.getStrokeColor")
def _roigetstroke(it, args, line):
    im = it.image(line)
    return (im.roi.stroke_color or "none") if im.roi is not None else "none"


@builtin("Roi.getFillColor")
def _roigetfill(it, args, line):
    im = it.image(line)
    return (im.roi.fill_color or "none") if im.roi is not None else "none"


@builtin("Roi.setStrokeWidth")
def _roiwidth(it, args, line):
    im = it.image(line)
    if im.roi is not None:
        im.roi.stroke_width = _n(args, 0)
        if hasattr(im.roi, "width") and im.roi.type == 5:
            im.roi.width = max(1.0, _n(args, 0))
        it.updated(im)


@builtin("Roi.getStrokeWidth")
def _roigetwidth(it, args, line):
    im = it.image(line)
    return float(im.roi.stroke_width) if im.roi is not None else 0.0


@builtin("Roi.setPosition")
def _roisetpos(it, args, line):
    im = it.image(line)
    if im.roi is None:
        raise MacroError("Selection required", line)
    if len(args) >= 3:
        im.roi.position = (int(_n(args, 0)), int(_n(args, 1)), int(_n(args, 2)))
    else:
        im.roi.position = (0, int(_n(args, 0)), 0)


@builtin("Roi.getPosition", refs=True)
def _roigetpos(it, nodes, line):
    im = it.image(line)
    pos = im.roi.position if im.roi is not None else (0, 0, 0)
    _out(it, nodes, [float(v) for v in pos], line)


@builtin("Roi.getBounds", refs=True)
def _roigetbounds(it, nodes, line):
    _selbounds(it, nodes, line)


@builtin("Roi.getCoordinates", refs=True)
def _roigetcoords(it, nodes, line):
    _selcoords(it, nodes, line)


@builtin("Roi.move")
def _roimove(it, args, line):
    im = it.image(line)
    if im.roi is not None:
        x, y, _, _ = im.roi.bounds()
        im.roi.translate(_n(args, 0) - x, _n(args, 1) - y)
        it.updated(im)


@builtin("Roi.remove")
def _roiremove(it, args, line):
    _select_none(it, "")


@builtin("Roi.size")
def _roisize(it, args, line):
    im = it.image(line)
    return float(len(im.roi.polygon()[0])) if im.roi is not None else 0.0


_COLOR_NAMES = {"black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#00ff00",
                "blue": "#0000ff", "yellow": "#ffff00", "cyan": "#00ffff",
                "magenta": "#ff00ff", "orange": "#ffc800", "pink": "#ffafaf",
                "gray": "#808080", "grey": "#808080", "lightgray": "#c0c0c0",
                "darkgray": "#404040"}


def _color_arg(args) -> str:
    if len(args) >= 3:
        r, g, b = (int(to_num(a)) & 255 for a in args[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    if not args:
        return "#ffff00"
    v = args[0]
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _COLOR_NAMES:
            return _COLOR_NAMES[s]
        if s.startswith("#"):
            return s
        if re.fullmatch(r"[0-9a-f]{6}|[0-9a-f]{8}", s):
            return "#" + s
        return s
    iv = int(to_num(v))
    return f"#{(iv >> 16) & 255:02x}{(iv >> 8) & 255:02x}{iv & 255:02x}"


# ---- overlay
@builtin("Overlay.addSelection")
def _ovadd(it, args, line):
    from rsstatchem.core.imaging.roi import Overlay
    im = it.image(line)
    if im.roi is None:
        raise MacroError("Selection required", line)
    roi = im.roi.copy()
    if args and _s(args, 0):
        roi.stroke_color = _color_arg([args[0]])
    if len(args) > 1:
        roi.stroke_width = _n(args, 1)
    if len(args) > 2 and _s(args, 2):
        roi.fill_color = _color_arg([args[2]])
    if im.overlay is None:
        im.overlay = Overlay()
    im.overlay.add(roi)
    it.updated(im)


@builtin("Overlay.size")
def _ovsize(it, args, line):
    im = it.session.active
    return float(len(im.overlay)) if im is not None and im.overlay is not None else 0.0


@builtin("Overlay.remove")
def _ovremove(it, args, line):
    im = it.image(line)
    im.overlay = None
    it.updated(im)


@builtin("Overlay.hidden")
def _ovhidden(it, args, line):
    im = it.image(line)
    return 1.0 if im.overlay is not None and im.overlay.hidden else 0.0


def _ov_vis(hidden):
    def fn(it, args, line):
        im = it.image(line)
        if im.overlay is not None:
            im.overlay.hidden = hidden
        it.updated(im)
    return fn


BUILTINS["Overlay.show"] = (_ov_vis(False), False)
BUILTINS["Overlay.hide"] = (_ov_vis(True), False)


@builtin("Overlay.activateSelection")
def _ovactivate(it, args, line):
    im = it.image(line)
    i = int(_n(args, 0))
    if im.overlay is None or not 0 <= i < len(im.overlay):
        raise MacroError("Index out of range", line)
    im.roi = im.overlay.rois[i].copy()
    it.updated(im)


# ---- ROI Manager
@builtin("roiManager")
def _roimanager(it, args, line):
    from rsstatchem.core.imaging.io.roi_codec import read_roi_zip, write_roi_zip
    s = it.session
    cmd = _s(args, 0).lower()
    rm = s.roi_manager
    sel = s.__dict__.setdefault("roi_manager_selected", [])
    if cmd == "add":
        im = it.image(line)
        if im.roi is None:
            raise MacroError("The active image does not have a selection.", line)
        roi = im.roi.copy()
        if im.stack_size > 1 and roi.position == (0, 0, 0):
            c, z, t = im.position
            roi.position = (c + 1, z + 1, t + 1) if (im.n_channels > 1 or im.n_frames > 1) \
                else (0, im.current_index + 1, 0)
        rm.append(roi)
    elif cmd == "count":
        return float(len(rm))
    elif cmd == "reset":
        rm.clear()
        sel.clear()
    elif cmd == "select":
        idx = args[1]
        indices = [int(to_num(v)) for v in idx] if isinstance(idx, list) else [int(to_num(idx))]
        sel[:] = indices
        if indices and s.active is not None and 0 <= indices[0] < len(rm):
            s.active.roi = rm[indices[0]].copy()
            it.updated(s.active)
    elif cmd == "deselect":
        sel.clear()
    elif cmd == "delete":
        for i in sorted(sel or range(len(rm)), reverse=True):
            if 0 <= i < len(rm):
                del rm[i]
        sel.clear()
    elif cmd == "index":
        return float(sel[0]) if sel else -1.0
    elif cmd == "rename":
        if sel:
            rm[sel[0]].name = _s(args, 1)
    elif cmd in ("save", "save selected"):
        write_roi_zip([rm[i] for i in sel] if cmd == "save selected" else rm, _s(args, 1))
    elif cmd == "open":
        rm.extend(read_roi_zip(_s(args, 1)) if _s(args, 1).lower().endswith(".zip") else
                  [__import__("rsstatchem.core.imaging.io.roi_codec",
                              fromlist=["read_roi"]).read_roi(_s(args, 1))])
    elif cmd == "measure":
        for i in (sel or range(len(rm))):
            s.active.roi = rm[i].copy()
            _run(it, ["Measure"], line)
    elif cmd in ("show all", "show none", "show all with labels", "show all without labels"):
        s.emit("roi_manager_show", cmd)
    elif cmd == "set color":
        for i in (sel or range(len(rm))):
            rm[i].stroke_color = _color_arg([args[1]])
    elif cmd == "set line width":
        for i in (sel or range(len(rm))):
            rm[i].stroke_width = _n(args, 1)
    else:
        if f"roiManager {cmd}" in s.registry:
            s.run(f"roiManager {cmd}", _s(args, 1, ""))
        else:
            raise MacroError(f'Unsupported roiManager command: "{cmd}"', line)
    s.emit("roi_manager_updated", rm)
    return None


@builtin("RoiManager.size")
def _rmsize(it, args, line):
    return float(len(it.session.roi_manager))


@builtin("RoiManager.select")
def _rmselect(it, args, line):
    return _roimanager(it, ["select", args[0]], line)


# ---- thresholds
@builtin("setThreshold")
def _setthreshold(it, args, line):
    from rsstatchem.core.imaging.image import Threshold
    im = it.image(line)
    lo, hi = _n(args, 0), _n(args, 1)
    mode = _s(args, 2, "red").lower()
    mode = {"red": "red", "black & white": "bw", "over/under": "over_under", "no color": "none",
            "raw": "red"}.get(mode, "red")
    cal = im.calibration
    if cal.calibrated() and im.bit_depth in (8, 16) and "raw" not in _s(args, 2, "").lower():
        lo, hi = cal.raw_value(lo, im.bit_depth), cal.raw_value(hi, im.bit_depth)
    im.threshold = Threshold(lo, hi, mode)
    it.updated(im)


@builtin("resetThreshold")
def _resetthreshold(it, args, line):
    im = it.image(line)
    im.threshold = None
    it.updated(im)


@builtin("getThreshold", refs=True)
def _getthreshold(it, nodes, line):
    im = it.image(line)
    t = im.threshold
    vals = [-1.0, -1.0] if t is None else [float(im.calibration.cvalue(t.lower)),
                                           float(im.calibration.cvalue(t.upper))]
    _out(it, nodes, vals, line)


@builtin("setAutoThreshold")
def _setautothreshold(it, args, line):
    """Delegates to the Process package (WP-B): ``threshold.set_auto_threshold(image, spec,
    options)`` when available, else a registered "setAutoThreshold" command."""
    im = it.image(line)
    spec = _s(args, 0, "Default")
    try:
        from rsstatchem.core.imaging import threshold as T
        fn = getattr(T, "set_auto_threshold", None)
    except ImportError:
        fn = None
    if fn is not None:
        try:
            fn(im, spec, mode="red")
        except ValueError as exc:
            raise MacroError(str(exc), line) from None
        it.updated(im)
        return None
    if "setAutoThreshold" in it.session.registry:
        it.session.run("setAutoThreshold", f"method=[{spec}]")
        return None
    raise MacroError("setAutoThreshold() needs the Process package (thresholds)", line)


# ---- measurements
def _stats_pixels(im, raw: bool = False):
    """Values inside the selection (or the whole plane) for getValue/getStatistics."""
    p = im.plane()
    if im.is_rgb:
        p = p.astype(np.float64)
        p = (p[..., 0] + p[..., 1] + p[..., 2]) / 3.0
    roi = im.roi
    mask = None
    if roi is not None and roi.is_area():
        mask = roi.get_mask((im.height, im.width))
        vals = p[mask]
    else:
        vals = p.ravel()
    vals = vals.astype(np.float64)
    if im.bit_depth == 32:
        vals = vals[np.isfinite(vals)]
    if not raw and im.calibration.calibrated() and not im.is_rgb:
        vals = np.asarray(im.calibration.cvalue(vals), float)
    return vals, mask


def measure_value(session, im, key: str) -> float:
    """``getValue(key)`` statistics computed directly (WP-C's ``measure.get_value`` is used
    instead when it exists)."""
    try:
        from rsstatchem.core.imaging import measure as M
        fn = getattr(M, "get_value", None)
    except ImportError:
        fn = None
    if fn is not None:
        return float(fn(session, im, key))
    raw = key.endswith(" raw")
    k = key[:-4] if raw else key
    cal = im.calibration
    if k == "Length":
        roi = im.roi
        if roi is None or not roi.is_line():
            return 0.0
        xs, ys = roi.polygon()
        return float(np.sum(np.hypot(np.diff(xs) * cal.pixel_width,
                                     np.diff(ys) * cal.pixel_height)))
    if k == "Angle":
        from rsstatchem.core.imaging.roi import LineRoi
        roi = im.roi
        if isinstance(roi, LineRoi):
            return float(roi.angle())
        if roi is not None and roi.type == 8:
            xs, ys = roi.polygon()
            a1 = math.atan2(ys[0] - ys[1], xs[0] - xs[1])
            a2 = math.atan2(ys[2] - ys[1], xs[2] - xs[1])
            d = abs(math.degrees(a1 - a2)) % 360
            return float(360 - d if d > 180 else d)
        return 0.0
    vals, mask = _stats_pixels(im, raw)
    n = vals.size
    if k == "Area":
        return float(n * cal.pixel_width * cal.pixel_height)
    if n == 0:
        return math.nan
    if k == "Mean":
        return float(vals.mean())
    if k == "Min":
        return float(vals.min())
    if k == "Max":
        return float(vals.max())
    if k == "StdDev":
        return float(vals.std(ddof=1)) if n > 1 else 0.0
    if k == "Median":
        return float(np.median(vals))
    if k == "IntDen":
        return float(vals.mean() * n * cal.pixel_width * cal.pixel_height)
    if k == "RawIntDen":
        rv, _ = _stats_pixels(im, True)
        return float(rv.sum())
    if k in ("X", "Y", "XM", "YM"):
        h, w = im.height, im.width
        yy, xx = np.mgrid[0:h, 0:w]
        if mask is None:
            mask = np.ones((h, w), bool)
        if k in ("X", "Y"):
            return float(((xx[mask] + 0.5).mean() * cal.pixel_width) if k == "X" else
                         ((yy[mask] + 0.5).mean() * cal.pixel_height))
        wts = im.plane()[mask].astype(np.float64)
        tot = wts.sum() or 1.0
        return float((((xx[mask] + 0.5) * wts).sum() / tot) * cal.pixel_width if k == "XM"
                     else (((yy[mask] + 0.5) * wts).sum() / tot) * cal.pixel_height)
    if k == "Skew":
        m, sd = vals.mean(), vals.std()
        return float(((vals - m) ** 3).mean() / sd ** 3) if sd else math.nan
    if k == "Kurt":
        m, sd = vals.mean(), vals.std()
        return float(((vals - m) ** 4).mean() / sd ** 4 - 3) if sd else math.nan
    if k == "Mode":
        if im.bit_depth == 8 or im.is_rgb:
            return float(np.bincount(vals.astype(int), minlength=256).argmax())
        u, c = np.unique(vals, return_counts=True)
        return float(u[c.argmax()])
    raise KeyError(key)


@builtin("getValue")
def _getvalue(it, args, line):
    s = it.session
    if len(args) == 3:
        im = it.image(line)
        return float(im.plane()[int(_n(args, 1)), int(_n(args, 0))])
    key = _s(args, 0)
    low = key.lower()
    if low == "results.count":
        return float(len(s.results))
    if low == "selection.size":
        im = s.active
        return float(len(im.roi.polygon()[0])) if im is not None and im.roi is not None \
            else 0.0
    if low == "selection.width":
        im = s.active
        return float(im.roi.stroke_width) if im is not None and im.roi is not None else 0.0
    if low in ("rgb.foreground", "rgb.background", "foreground.color",
               "background.color"):
        c = s.options.foreground if "fore" in low else s.options.background
        return float(int(c.lstrip("#")[-6:], 16))
    if low == "font.size":
        return float(s.options.extra.get("font.size", 18))
    if low == "line.width":
        return float(s.options.line_width)
    if low == "image.size":
        return float(it.image(line).size_bytes())
    im = it.image(line)
    try:
        return measure_value(s, im, key)
    except KeyError:
        raise MacroError(f'Invalid key: "{key}"', line) from None


def _statistics(it, nodes, line, raw):
    im = it.image(line)
    vals, mask = _stats_pixels(im, raw)
    cal = im.calibration
    n = vals.size
    out = [float(n * cal.pixel_width * cal.pixel_height), float(vals.mean()) if n else 0.0,
           float(vals.min()) if n else 0.0, float(vals.max()) if n else 0.0,
           float(vals.std(ddof=1)) if n > 1 else 0.0]
    if len(nodes) > 5:
        if im.bit_depth in (8, 24):
            hist = np.bincount(np.clip(vals, 0, 255).astype(int), minlength=256)[:256]
        else:
            hist, _ = np.histogram(vals, 256)
        out.append([float(v) for v in hist])
    _out(it, nodes, out[:len(nodes)], line)


@builtin("getStatistics", refs=True)
def _getstatistics(it, nodes, line):
    _statistics(it, nodes, line, False)


@builtin("getRawStatistics", refs=True)
def _getrawstatistics(it, nodes, line):
    _statistics(it, nodes, line, True)


@builtin("getHistogram", refs=True)
def _gethistogram(it, nodes, line):
    im = it.image(line)
    nbins = int(to_num(it.eval(nodes[2])))
    vals, _ = _stats_pixels(im, True)
    if im.bit_depth in (8, 24) and nbins == 256:
        counts = np.bincount(vals.astype(int), minlength=256)[:256]
        values = np.arange(256, dtype=float)
    elif im.bit_depth == 16 and nbins == 65536:
        counts = np.bincount(vals.astype(int), minlength=65536)
        values = np.arange(65536, dtype=float)
    else:
        lo, hi = (vals.min(), vals.max()) if vals.size else (0, 1)
        counts, edges = np.histogram(vals, nbins, (lo, hi))
        values = edges[:-1]
    _out(it, nodes[:2], [[float(v) for v in values], [float(c) for c in counts]], line)


@builtin("List.setMeasurements")
def _list_setmeasurements(it, args, line):
    im = it.image(line)
    d = it.session.__dict__.setdefault("macro_list", {})
    for key in ("Area", "Mean", "StdDev", "Min", "Max", "Median", "X", "Y", "IntDen",
                "RawIntDen", "Skew", "Kurt"):
        try:
            d[key] = n2s(measure_value(it.session, im, key))
        except (KeyError, ValueError):
            pass


@builtin("List.get")
def _list_get(it, args, line):
    return it.session.__dict__.setdefault("macro_list", {}).get(_s(args, 0), "")


@builtin("List.getValue")
def _list_getvalue(it, args, line):
    v = it.session.__dict__.setdefault("macro_list", {}).get(_s(args, 0), "")
    try:
        return float(v)
    except ValueError:
        return math.nan


@builtin("List.set")
def _list_set(it, args, line):
    it.session.__dict__.setdefault("macro_list", {})[_s(args, 0)] = _s(args, 1)


@builtin("List.clear")
def _list_clear(it, args, line):
    it.session.__dict__["macro_list"] = {}


# ---- results tables
@builtin("nResults")
def _nresults(it, args, line):
    return float(len(it.session.results))


def _table_for(it, args, i):
    title = _s(args, i, "") if i < len(args) else ""
    return it.session.table(title) if title else it.session.results


@builtin("getResult")
def _getresult(it, args, line):
    rt = _table_for(it, args, 2)
    col = _s(args, 0)
    row = int(_n(args, 1, len(rt) - 1))
    if not rt.column_exists(col):
        return math.nan
    if not 0 <= row < len(rt):
        raise MacroError(f"Row ({row}) out of range", line)
    v = rt.get_value(col, row)
    return v if isinstance(v, str) else float(v)


@builtin("getResultString")
def _getresultstring(it, args, line):
    rt = _table_for(it, args, 2)
    v = rt.get_value(_s(args, 0), int(_n(args, 1, len(rt) - 1)))
    return v if isinstance(v, str) else n2s(v)


@builtin("getResultLabel")
def _getresultlabel(it, args, line):
    return it.session.results.get_label(int(_n(args, 0))) or ""


@builtin("setResult")
def _setresult(it, args, line):
    rt = _table_for(it, args, 3)
    col, row = _s(args, 0), int(_n(args, 1))
    if row == len(rt):
        rt.increment_counter()
    if col == "Label":
        rt.set_label(_s(args, 2), row)
    else:
        v = args[2]
        rt.set_value(col, row, v if isinstance(v, str) else to_num(v))
    return None


@builtin("updateResults")
def _updateresults(it, args, line):
    it.session.show_table(it.session.results)


@builtin("Table.size")
def _tsize(it, args, line):
    return float(len(_table_for(it, args, 0)))


@builtin("Table.get", "Table.getString")
def _tget(it, args, line):
    rt = _table_for(it, args, 2)
    v = rt.get_value(_s(args, 0), int(_n(args, 1)))
    return v if isinstance(v, str) else float(v)


@builtin("Table.set")
def _tset(it, args, line):
    rt = _table_for(it, args, 3)
    row = int(_n(args, 1))
    if row == len(rt):
        rt.increment_counter()
    v = args[2]
    rt.set_value(_s(args, 0), row, v if isinstance(v, str) else to_num(v))


@builtin("Table.getColumn")
def _tgetcol(it, args, line):
    rt = _table_for(it, args, 1)
    return [float(v) if not isinstance(v, str) else v for v in rt.column(_s(args, 0))]


@builtin("Table.setColumn")
def _tsetcol(it, args, line):
    rt = _table_for(it, args, 2)
    vals = args[1] if len(args) > 1 else []
    while len(rt) < len(vals):
        rt.increment_counter()
    for i, v in enumerate(vals):
        rt.set_value(_s(args, 0), i, v if isinstance(v, str) else to_num(v))


@builtin("Table.headings")
def _theadings(it, args, line):
    return "\t".join(_table_for(it, args, 0).headings)


@builtin("Table.create", "Table.reset")
def _tcreate(it, args, line):
    from rsstatchem.core.imaging.results import ResultsTable
    title = _s(args, 0, "Results")
    rt = ResultsTable.results(title) if title == "Results" else ResultsTable(title)
    it.session.show_table(rt)


@builtin("Table.update")
def _tupdate(it, args, line):
    it.session.show_table(_table_for(it, args, 0))


@builtin("Table.deleteRows")
def _tdeleterows(it, args, line):
    rt = _table_for(it, args, 2)
    for i in range(int(_n(args, 1)), int(_n(args, 0)) - 1, -1):
        rt.delete_row(i)


@builtin("String.getResultsHeadings")
def _sheadings(it, args, line):
    return "\t".join(it.session.results.headings)


# ---- colours and drawing
@builtin("setForegroundColor")
def _setfg(it, args, line):
    it.session.options.foreground = _color_arg(args)


@builtin("setBackgroundColor")
def _setbg(it, args, line):
    it.session.options.background = _color_arg(args)


@builtin("setColor")
def _setcolor(it, args, line):
    if len(args) == 1 and not isinstance(args[0], str):
        it.colors["foreground"] = ("value", to_num(args[0]))
    else:
        it.colors["foreground"] = ("rgb", _color_arg(args))


@builtin("setLineWidth")
def _setlinewidth(it, args, line):
    it.session.options.line_width = _n(args, 0)


def _draw_value(it, im):
    from rsstatchem.ui.imagej import commands_edit as E  # noqa: F401 (pure helpers)
    c = it.colors.get("foreground")
    if c is None:
        return E.color_value(im, it.session.options.foreground)
    if c[0] == "value":
        return c[1] if not im.is_rgb else E.color_value(im, f"#{int(c[1]) & 0xFFFFFF:06x}")
    return E.color_value(im, c[1])


def _shape_fn(kind):
    def fn(it, args, line):
        from rsstatchem.core.imaging.roi import OvalRoi, RectRoi
        from rsstatchem.ui.imagej import commands_edit as E
        im = it.image(line)
        x, y, w, h = (_n(args, i) for i in range(4))
        roi = RectRoi(x, y, w, h) if "Rect" in kind else OvalRoi(x, y, w, h)
        value = _draw_value(it, im)
        if kind.startswith("fill"):
            E.fill_mask(im, roi.get_mask((im.height, im.width)), value)
        else:
            E.draw_roi_outline(im, roi, value, 1)
        im.changed = True
    return fn


for _k in ("fillRect", "fillOval", "drawRect", "drawOval"):
    BUILTINS[_k] = (_shape_fn(_k), False)


@builtin("drawLine")
def _drawline(it, args, line):
    from rsstatchem.core.imaging.roi import LineRoi
    from rsstatchem.ui.imagej import commands_edit as E
    im = it.image(line)
    roi = LineRoi(*(_n(args, i) for i in range(4)))
    E.draw_roi_outline(im, roi, _draw_value(it, im), int(it.session.options.line_width))
    im.changed = True


@builtin("fill")
def _fill(it, args, line):
    from rsstatchem.ui.imagej import commands_edit as E
    im = it.image(line)
    mask = im.roi.get_mask((im.height, im.width)) if im.roi is not None and im.roi.is_area() \
        else np.ones((im.height, im.width), bool)
    E.fill_mask(im, mask, _draw_value(it, im))
    im.changed = True
    it.updated(im)


@builtin("setTool")
def _settool(it, args, line):
    if it.ui.set_tool is not None:
        it.ui.set_tool(args[0] if args else 0)


@builtin("doCommand")
def _docommand(it, args, line):
    return _run(it, args[:1], line)


# ---- files
@builtin("File.exists")
def _fexists(it, args, line):
    return 1.0 if Path(_s(args, 0)).exists() else 0.0


@builtin("File.isDirectory")
def _fisdir(it, args, line):
    return 1.0 if Path(_s(args, 0)).is_dir() else 0.0


@builtin("File.getName")
def _fname(it, args, line):
    return Path(_s(args, 0)).name


@builtin("File.getNameWithoutExtension", "File.nameWithoutExtension")
def _fstem(it, args, line):
    if not args:
        im = it.session.active
        return Path(im.title).stem if im is not None else ""
    return Path(_s(args, 0)).stem


@builtin("File.getDirectory", "File.getParent")
def _fdir(it, args, line):
    return str(Path(_s(args, 0)).parent) + "/"


@builtin("File.makeDirectory")
def _fmkdir(it, args, line):
    Path(_s(args, 0)).mkdir(parents=True, exist_ok=True)


@builtin("File.delete")
def _fdelete(it, args, line):
    p = Path(_s(args, 0))
    try:
        if p.is_dir():
            p.rmdir()
        else:
            p.unlink()
        return 1.0
    except OSError:
        return 0.0


@builtin("File.length")
def _flength(it, args, line):
    p = Path(_s(args, 0))
    return float(p.stat().st_size) if p.exists() else 0.0


@builtin("File.openAsString", "File.openAsRawString")
def _fopenstr(it, args, line):
    return Path(_s(args, 0)).read_text(errors="replace")


@builtin("File.saveString")
def _fsavestr(it, args, line):
    Path(_s(args, 1)).write_text(_s(args, 0))


@builtin("File.append")
def _fappend(it, args, line):
    with open(_s(args, 1), "a", encoding="utf-8") as fh:
        fh.write(_s(args, 0) + "\n")


@builtin("File.open")
def _fopen(it, args, line):
    path = _s(args, 0)
    return {"file": path, "lines": []}


@builtin("File.close")
def _fclose(it, args, line):
    fh = args[0]
    if isinstance(fh, dict) and "file" in fh:
        Path(fh["file"]).write_text("\n".join(fh["lines"]) + ("\n" if fh["lines"] else ""),
                                    encoding="utf-8")


@builtin("getFileList")
def _getfilelist(it, args, line):
    d = Path(_s(args, 0))
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir(), key=lambda q: q.name):
        if p.name.startswith("."):
            continue
        out.append(p.name + ("/" if p.is_dir() else ""))
    return out


@builtin("getDirectory", "getDir", "File.directory")
def _getdirectory(it, args, line):
    key = _s(args, 0, "").lower()
    home = Path.home()
    if key in ("home",):
        return str(home) + "/"
    if key in ("temp",):
        import tempfile
        return tempfile.gettempdir() + "/"
    if key in ("image",):
        im = it.session.active
        return str(Path(im.path).parent) + "/" if im is not None and im.path else ""
    if key in ("file",) and it.path:
        return str(Path(it.path).parent) + "/"
    if key in ("startup", "imagej", "plugins", "macros", "luts"):
        return str(home) + "/"
    if it.ui.get_string is not None and key not in ("cwd",):
        v = it.ui.get_string("Choose a Directory", "")
        if v is None:
            raise MacroExit("")
        return v
    return str(Path.cwd()) + "/"


CONSTANTS["File.separator"] = "/"


@builtin("File.separator")
def _fsep(it, args, line):
    return "/"


@builtin("File.name", "File.nameWithoutExtension")
def _fcurname(it, args, line):
    im = it.session.active
    return "" if im is None else Path(im.title).name


# ---------------------------------------------------------------------------------- API
def run_macro(session, code: str, argument: str = "", ui: UIHooks | None = None,
              path: str | None = None, macro_name: str | None = None):
    """Run ImageJ macro ``code``; returns the value of a top-level ``return`` (or None).
    Errors raise :class:`MacroError`; ``exit`` ends quietly.

    Batch mode (Process > Batch, ``Interpreter.runBatchMacro``): when the third argument is
    an ImageDoc, the macro runs on that image (made active temporarily, it need not be open)
    and the image active at the end is returned; images the macro opened are closed again."""
    from rsstatchem.core.imaging.image import ImageDoc
    if isinstance(argument, ImageDoc):
        return _run_batch(session, code, argument, ui, path)
    prog = parse(code)
    it = Interpreter(session, ui, argument, path)
    try:
        return it.run_program(prog, macro_name)
    except MacroExit:
        return None
    except RecursionError:
        raise MacroError("Stack overflow", it.line) from None


def _run_batch(session, code: str, image, ui=None, path=None):
    old_active = session.active
    before = list(session.images)
    session.active = image
    try:
        prog = parse(code)
        it = Interpreter(session, ui, "", path)
        it.batch_mode = True
        try:
            it.run_program(prog)
        except MacroExit:
            pass
        result = session.active if session.active is not None else image
    finally:
        for im in list(session.images):
            if im not in before:
                session.close_image(im)
        session.active = old_active if old_active in session.images else \
            (session.images[-1] if session.images else None)
    return result


def macro_names(code: str) -> list[str]:
    """Names of the ``macro "..." {}`` blocks (Plugins > Macros > Install...)."""
    return [name for name, _ in parse(code).macros]


# ---------------------------------------------------------------------------------- Python
class ScriptAPI:
    """The ``ij`` object of Python scripts: every macro built-in as a method
    (``ij.run("Invert")``, ``ij.makeOval(1, 2, 3, 4)``); ``ij.run`` also accepts keyword
    options like the Python recorder writes (``ij.run("Gaussian Blur...", sigma=2,
    stack=True)``). Functions with output arguments return tuples."""

    def __init__(self, session, ui: UIHooks | None = None):
        self._it = Interpreter(session, ui)
        self.session = session

    @property
    def image(self):
        return self.session.active

    def run(self, label: str, options: str = "", **kw):
        from rsstatchem.core.imaging.commands import format_number
        parts = [options] if options else []
        for k, v in kw.items():
            if v is True:
                parts.append(k)
            elif v is False or v is None:
                continue
            elif isinstance(v, (int, float)):
                parts.append(f"{k}={format_number(v, 4)}")
            else:
                s = str(v)
                parts.append(f"{k}=[{s}]" if " " in s or not s else f"{k}={s}")
        return _run(self._it, [label, " ".join(parts)], 0)

    def __getattr__(self, name: str):
        key = name.replace("_", ".", 1) if name.split("_")[0] in NAMESPACES else name
        spec = BUILTINS.get(key)
        if spec is None:
            raise AttributeError(name)
        fn, refs = spec

        def call(*args):
            vals = [_py_value(a) for a in args]
            if refs:
                # output-argument function: return the outputs as a tuple
                n_in = {"getHistogram": 3, "Array.getStatistics": 1}.get(key, 0)
                names = [f"__o{i}" for i in range(8)]
                frame = self._it.frames[-1]
                nodes = []
                for i in range(n_in):
                    frame[f"__in{i}"] = vals[i]
                    nodes.append(("var", 0, f"__in{i}"))
                nodes += [("var", 0, n) for n in names]
                fn(self._it, nodes, 0)
                outs = [frame.pop(n) for n in names if n in frame]
                return tuple(outs)
            return fn(self._it, vals, 0)
        return call


def _py_value(a):
    if isinstance(a, bool):
        return 1.0 if a else 0.0
    if isinstance(a, (list, tuple, np.ndarray)):
        return [_py_value(v) for v in a]
    if isinstance(a, (int, float, np.integer, np.floating)):
        return float(a)
    return str(a)


def run_python(session, code: str, ui: UIHooks | None = None, path: str | None = None):
    """Run a Python script with ``ij`` (ScriptAPI), ``session``, ``image`` and ``np``."""
    api = ScriptAPI(session, ui)
    env = {"ij": api, "session": session, "image": session.active, "np": np,
           "__name__": "__rsij_script__", "__file__": path or "<script>"}
    exec(compile(code, path or "<script>", "exec"), env)  # noqa: S102 - user's own script
    return env.get("result")


# ---------------------------------------------------------------------------------- recording
def _js(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt(v) -> str:
    if isinstance(v, str):
        return _js(v)
    return n2s(v)


def call_line(language: str, name: str, *args) -> str:
    """One recorded statement, e.g. ``makeRectangle(1, 2, 3, 4);`` or
    ``ij.makeRectangle(1, 2, 3, 4)``."""
    if language == "Python":
        py = []
        for a in args:
            py.append(repr(a) if isinstance(a, str) else n2s(a))
        return f"ij.{name.replace('.', '_')}({', '.join(py)})"
    return f"{name}({', '.join(_fmt(a) for a in args)});"


def roi_call(roi) -> tuple[str, tuple] | None:
    """The macro call that recreates ``roi`` (what the Recorder writes for tool selections)."""
    from rsstatchem.core.imaging.roi import shapes as S

    if isinstance(roi, S.TextRoi):
        return "makeText", (roi.text, roi.x, roi.y)
    if isinstance(roi, S.RectRoi):
        args = (roi.x, roi.y, roi.width, roi.height)
        if roi.corner_diameter:
            args += (roi.corner_diameter,)
        return "makeRectangle", args
    if isinstance(roi, S.OvalRoi):
        return "makeOval", (roi.x, roi.y, roi.width, roi.height)
    if isinstance(roi, S.EllipseRoi):
        return "makeEllipse", (roi.x1, roi.y1, roi.x2, roi.y2, roi.aspect_ratio)
    if isinstance(roi, S.RotatedRectRoi):
        return "makeRotatedRectangle", (roi.x1, roi.y1, roi.x2, roi.y2, roi.rect_width)
    if isinstance(roi, S.LineRoi):
        if roi.arrow:
            return "makeArrow", (roi.x1, roi.y1, roi.x2, roi.y2, "filled")
        return "makeLine", (roi.x1, roi.y1, roi.x2, roi.y2)
    if isinstance(roi, S.PointRoi):
        xs, ys = roi.polygon()
        if len(xs) == 1:
            return "makePoint", (float(xs[0]), float(ys[0]))
        return "makeSelection", ("point",) + _coords(xs, ys)
    if isinstance(roi, S.PolygonRoi):
        xs, ys = roi.polygon()
        if roi.type == S.POLYGON and np.all(np.mod(xs, 1) == 0) and np.all(np.mod(ys, 1) == 0):
            flat = []
            for x, y in zip(xs.tolist(), ys.tolist()):
                flat += [x, y]
            return "makePolygon", tuple(flat)
        if roi.type == S.POLYLINE and np.all(np.mod(xs, 1) == 0):
            flat = []
            for x, y in zip(xs.tolist(), ys.tolist()):
                flat += [x, y]
            return "makeLine", tuple(flat)
        kind = {S.POLYGON: "polygon", S.FREEROI: "freehand", S.TRACED_ROI: "traced",
                S.POLYLINE: "polyline", S.FREELINE: "freeline", S.ANGLE: "angle"}[roi.type]
        return "makeSelection", (kind,) + _coords(xs, ys)
    return None


class _Arr(list):
    """Marks an array argument in recorded calls (written as newArray(...))."""


def _coords(xs, ys):
    return (_Arr(float(v) for v in xs), _Arr(float(v) for v in ys))


def roi_line(language: str, roi) -> str | None:
    rc = roi_call(roi)
    if rc is None:
        return None
    name, args = rc
    if language == "Python":
        parts = []
        for a in args:
            if isinstance(a, _Arr):
                parts.append("[" + ", ".join(n2s(v) for v in a) + "]")
            else:
                parts.append(repr(a) if isinstance(a, str) else n2s(a))
        return f"ij.{name}({', '.join(parts)})"
    parts = []
    for a in args:
        if isinstance(a, _Arr):
            parts.append("newArray(" + ",".join(n2s(v) for v in a) + ")")
        else:
            parts.append(_fmt(a))
    return f"{name}({', '.join(parts)});"

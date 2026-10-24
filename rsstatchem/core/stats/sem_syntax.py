"""lavaan-style model syntax for covariance-based SEM (CB-SEM), mirroring R lavaan / AMOS.

Supported syntax (one statement per line, ``;`` also separates, ``#`` starts a comment, a line
ending in ``+`` or an operator continues on the next line):

- ``f =~ x1 + x2 + x3``   measurement (loadings); the first loading is fixed to 1 (marker)
- ``y ~ x1 + f``          regressions
- ``x1 ~~ x2``            (residual) covariances, ``x1 ~~ x1`` a (residual) variance
- modifiers ``1*x`` (fix), ``NA*x`` (free), ``a*x`` (label; shared labels = equality),
  ``start(0.5)*x``, several at once ``NA*a*x``
- ``ind := a*b``          defined parameters (arithmetic of labels, sqrt/exp/log/abs)
- ``a == b``              equality constraint between labels (``a == 0.5`` fixes)

The parser builds the full parameter table with lavaan ``sem()`` defaults: residual variances
for every variable, covariances among exogenous variables, residual covariances among pure
outcomes (``auto.cov.y``), and identification by marker variable or by standardised latent
variances (``std.lv``). No Qt here.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

OPERATORS = (":=", "=~", "~~", "==", "~", "<", ">")
_IDENT = re.compile(r"^[A-Za-z_.][A-Za-z0-9_.]*$")
_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "abs": abs}


class SEMSyntaxError(ValueError):
    """Raised with a list of ``(line, message)`` issues."""

    def __init__(self, issues: list[tuple[int, str]]):
        self.issues = issues
        super().__init__("\n".join(f"Line {ln}: {msg}" if ln else msg for ln, msg in issues))


@dataclass
class Term:
    name: str
    fixed: float | None = None
    free: bool = False          # explicit NA*
    label: str | None = None
    start: float | None = None


@dataclass
class Statement:
    line: int
    lhs: str
    op: str
    terms: list[Term]


@dataclass
class Defined:
    line: int
    name: str
    expr: str
    names: list[str]
    code: object = None

    def evaluate(self, env: dict):
        return eval(self.code, {"__builtins__": {}}, {**_NP_FUNCS, **env})  # noqa: S307


@dataclass
class Param:
    lhs: str
    op: str                     # "=~", "~", "~~"
    rhs: str
    free: bool = True
    value: float = 0.0          # fixed value, or start value if free
    label: str | None = None
    user: bool = True
    start: float | None = None
    index: int = -1             # free-parameter index (shared by equal labels)
    line: int = 0
    explicit: bool = False      # user fixed (number) or freed (NA*) it

    @property
    def name(self) -> str:
        return f"{self.lhs}{self.op}{self.rhs}"


@dataclass
class SEMModel:
    params: list[Param]
    observed: list[str]         # in model order (dataset spelling)
    latent: list[str]
    defined: list[Defined]
    indicators: dict[str, list[str]]
    endogenous: list[str]
    exogenous: list[str]
    n_free: int
    warnings: list[str] = field(default_factory=list)
    syntax: str = ""
    std_lv: bool = False

    @property
    def variables(self) -> list[str]:
        return self.observed + self.latent

    def labels(self) -> dict[str, int]:
        return {p.label: p.index for p in self.params if p.label and p.free}

    def structural(self) -> list[str]:
        """Latents plus observed variables that are not indicators (path-model nodes)."""
        ind = {x for xs in self.indicators.values() for x in xs}
        return [v for v in self.variables if v in self.latent or v not in ind]


def _np_funcs():
    import numpy as np
    return {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs}


_NP_FUNCS = _np_funcs()


# ------------------------------------------------------------------ lexing
def _logical_lines(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    pending, start = "", 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = re.split(r"[#!]", raw, maxsplit=1)[0]
        for part in line.split(";"):
            part = part.strip()
            if not part:
                continue
            if pending and (pending.endswith(("+", "*", "~", "=", ",")) or part.startswith("+")
                            or not any(op in part for op in OPERATORS)):
                pending = f"{pending} {part}"
                continue
            if pending:
                out.append((start, pending))
            pending, start = part, number
    if pending:
        out.append((start, pending))
    return out


def _split_op(text: str) -> tuple[str, str, str] | None:
    for op in OPERATORS:
        i = text.find(op)
        if i < 0:
            continue
        if op == "~" and (text[i:i + 2] == "~~"):
            continue
        if op in ("<", ">") and "==" in text:
            continue
        return text[:i].strip(), op, text[i + len(op):].strip()
    return None


def _parse_number(tok: str) -> float | None:
    tok = tok.strip()
    return float(tok) if _NUMBER.match(tok) else None


def _parse_term(raw: str, line: int, issues: list) -> Term | None:
    parts = [p.strip() for p in raw.split("*")]
    name = parts[-1]
    if not name:
        issues.append((line, f"Missing variable name after '*' in '{raw.strip()}'."))
        return None
    if _parse_number(name) is not None and name.strip() in ("1", "0") and len(parts) <= 2:
        return Term(name="1")  # intercept term, rejected by the caller
    if not _IDENT.match(name):
        issues.append((line, f"'{name}' is not a valid variable name."))
        return None
    term = Term(name=name)
    for mod in parts[:-1]:
        if not mod:
            issues.append((line, f"Empty modifier in '{raw.strip()}'."))
            return None
        num = _parse_number(mod)
        if num is not None:
            term.fixed = num
        elif mod.upper() == "NA":
            term.free = True
        elif m := re.fullmatch(r"start\(\s*([^)]+)\s*\)", mod):
            val = _parse_number(m.group(1))
            if val is None:
                issues.append((line, f"start() needs a number: '{mod}'."))
                return None
            term.start = val
        elif m := re.fullmatch(r"(?:label|equal)\(\s*['\"]?([A-Za-z_.][\w.]*)['\"]?\s*\)", mod):
            term.label = m.group(1)
        elif _IDENT.match(mod):
            term.label = mod
        else:
            issues.append((line, f"Unknown modifier '{mod}' (use a number, NA, a label or "
                                 f"start(value))."))
            return None
    return term


def parse(text: str) -> tuple[list[Statement], list[Defined], list[tuple[int, str, str]],
                              list[tuple[int, str]]]:
    """Statements, defined parameters, equality constraints and issues (line, message)."""
    statements: list[Statement] = []
    defined: list[Defined] = []
    constraints: list[tuple[int, str, str]] = []
    issues: list[tuple[int, str]] = []
    for line, body in _logical_lines(text):
        split = _split_op(body)
        if split is None:
            issues.append((line, f"No operator found in '{body}' (expected =~, ~, ~~ or :=)."))
            continue
        lhs, op, rhs = split
        if op in ("<", ">"):
            issues.append((line, "Inequality constraints (<, >) are not supported."))
            continue
        if not lhs or not rhs:
            issues.append((line, f"Incomplete statement '{body}'."))
            continue
        if op == ":=":
            if not _IDENT.match(lhs):
                issues.append((line, f"'{lhs}' is not a valid name for a defined parameter."))
                continue
            d = _parse_defined(line, lhs, rhs, issues)
            if d:
                defined.append(d)
            continue
        if op == "==":
            constraints.append((line, lhs, rhs))
            continue
        lhs_names = [s.strip() for s in lhs.split("+")]
        bad = [n for n in lhs_names if not _IDENT.match(n)]
        if bad:
            issues.append((line, f"'{bad[0]}' is not a valid variable name (left of {op})."))
            continue
        terms = []
        for raw in rhs.split("+"):
            if not raw.strip():
                issues.append((line, f"Empty term in '{body}'."))
                continue
            t = _parse_term(raw, line, issues)
            if t is None:
                continue
            if t.name == "1":
                issues.append((line, "Intercepts / mean structure (~ 1) are not supported; "
                                     "CB-SEM here analyses the covariance matrix only."))
                continue
            terms.append(t)
        if not terms:
            continue
        for name in lhs_names:
            statements.append(Statement(line, name, op, [Term(**vars(t)) for t in terms]))
    return statements, defined, constraints, issues


def _parse_defined(line: int, name: str, expr: str, issues: list) -> Defined | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        issues.append((line, f"Cannot parse the expression '{expr}'."))
        return None
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.operator,
                             ast.unaryop, ast.Load, ast.Constant)):
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                issues.append((line, f"Only numbers are allowed as constants in '{expr}'."))
                return None
            continue
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS) or node.keywords:
                issues.append((line, f"Only sqrt(), exp(), log() and abs() may be used in "
                                     f"'{expr}'."))
                return None
            continue
        if isinstance(node, ast.Name):
            if node.id not in _FUNCS:
                names.append(node.id)
            continue
        issues.append((line, f"Unsupported element in the expression '{expr}'."))
        return None
    return Defined(line, name, expr, list(dict.fromkeys(names)),
                   compile(tree, "<defined>", "eval"))


# ------------------------------------------------------------------ model building
def build_model(text: str, available: list[str] | None = None, std_lv: bool = False,
                auto_cov_y: bool = True, cov_exogenous: bool = True,
                numeric: set[str] | None = None) -> SEMModel:
    """Parse ``text`` and build the full parameter table. ``available`` = dataset variable
    names (matched case-insensitively); raises :class:`SEMSyntaxError` on problems."""
    statements, defined, constraints, issues = parse(text)
    if not statements and not issues:
        issues.append((0, "The model is empty. Specify at least one =~ or ~ statement."))
    if issues:
        raise SEMSyntaxError(issues)
    lookup = {n.lower(): n for n in (available or [])}

    latent: list[str] = []
    for s in statements:
        if s.op == "=~" and s.lhs not in latent:
            latent.append(s.lhs)
    latent_l = {v.lower(): v for v in latent}

    def canon(name: str, line: int) -> str:
        if name in latent:
            return name
        if name.lower() in latent_l:
            return latent_l[name.lower()]
        if available is None:
            return name
        if name.lower() in lookup:
            return lookup[name.lower()]
        issues.append((line, f"'{name}' is not a variable in the dataset and is not defined as "
                             f"a latent variable (with =~)."))
        return name

    for v in latent:
        if available is not None and v.lower() in lookup:
            issues.append((0, f"The latent variable name '{v}' is also an observed variable in "
                              f"the dataset; choose another name."))
    for s in statements:
        s.lhs = canon(s.lhs, s.line)
        for t in s.terms:
            t.name = canon(t.name, s.line)
        if s.op == "=~":
            for t in s.terms:
                if t.name == s.lhs:
                    issues.append((s.line, f"'{s.lhs}' cannot be an indicator of itself."))
        if s.op == "~":
            for t in s.terms:
                if t.name == s.lhs:
                    issues.append((s.line, f"'{s.lhs}' cannot be regressed on itself."))
    if issues:
        raise SEMSyntaxError(issues)

    order: list[str] = []
    for s in statements:
        for v in [s.lhs] + [t.name for t in s.terms]:
            if v not in order:
                order.append(v)
    observed = [v for v in order if v not in latent]
    if numeric is not None:
        for v in observed:
            if v not in numeric:
                issues.append((0, f"'{v}' is not a numeric variable."))
    if issues:
        raise SEMSyntaxError(issues)

    indicators: dict[str, list[str]] = {f: [] for f in latent}
    regressed: set[str] = set()
    predictors: set[str] = set()
    for s in statements:
        if s.op == "=~":
            for t in s.terms:
                if t.name not in indicators[s.lhs]:
                    indicators[s.lhs].append(t.name)
        elif s.op == "~":
            regressed.add(s.lhs)
            predictors.update(t.name for t in s.terms)
    for f, xs in indicators.items():
        if not xs:
            issues.append((0, f"Latent variable '{f}' has no indicators."))
    is_indicator = {x for xs in indicators.values() for x in xs}
    endogenous = [v for v in order if v in is_indicator or v in regressed]
    exogenous = [v for v in order if v not in endogenous]

    params: list[Param] = []
    by_key: dict[tuple[str, str, str], Param] = {}

    def key(lhs: str, op: str, rhs: str) -> tuple[str, str, str]:
        if op == "~~":
            a, b = sorted((lhs, rhs), key=order.index)
            return a, op, b
        return lhs, op, rhs

    def add(lhs, op, rhs, line=0, user=True, term: Term | None = None, free=True, value=0.0):
        k = key(lhs, op, rhs)
        p = by_key.get(k)
        if p is None:
            p = Param(k[0], op, k[2], free=free, value=value, user=user, line=line)
            by_key[k] = p
            params.append(p)
        if term is not None:
            if term.fixed is not None:
                p.free, p.value = False, term.fixed
            if term.free:
                p.free = True
            if term.label:
                p.label = term.label
            if term.start is not None:
                p.start = term.start
            p.user = True
            p.explicit = p.explicit or term.fixed is not None or term.free
        return p

    for s in statements:
        for t in s.terms:
            if s.op == "=~":
                add(s.lhs, "=~", t.name, s.line, term=t)
            elif s.op == "~":
                add(s.lhs, "~", t.name, s.line, term=t)
            else:
                add(s.lhs, "~~", t.name, s.line, term=t)

    # identification of latent scales
    for f in latent:
        loads = [p for p in params if p.op == "=~" and p.lhs == f]
        var_key = (f, "~~", f)
        user_var = by_key.get(var_key)
        if std_lv:
            if user_var is None:
                add(f, "~~", f, user=False, free=False, value=1.0)
        else:
            if loads and not any(not p.free for p in loads):
                first = loads[0]
                if not first.explicit:
                    first.free, first.value = False, 1.0
    # residual / exogenous variances
    for v in order:
        if (v, "~~", v) not in by_key:
            add(v, "~~", v, user=False)
    # covariances among exogenous variables
    if cov_exogenous:
        exo = [v for v in exogenous]
        for i, a in enumerate(exo):
            for b in exo[i + 1:]:
                if key(a, "~~", b) not in by_key:
                    add(a, "~~", b, user=False)
    # residual covariances among pure outcomes (lavaan auto.cov.y)
    if auto_cov_y:
        pure = [v for v in order if v in regressed and v not in predictors
                and v not in is_indicator]
        for i, a in enumerate(pure):
            for b in pure[i + 1:]:
                if key(a, "~~", b) not in by_key:
                    add(a, "~~", b, user=False)

    # equality constraints by label (union of labels), then free indices
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    fixed_labels: dict[str, float] = {}
    label_names = {p.label for p in params if p.label}
    for line, a, b in constraints:
        num = _parse_number(b)
        if a not in label_names:
            issues.append((line, f"Unknown label '{a}' in constraint."))
            continue
        if num is not None:
            fixed_labels[a] = num
        elif b in label_names:
            parent[find(a)] = find(b)
        else:
            issues.append((line, f"Unknown label '{b}' in constraint."))
    for p in params:
        if p.label and (p.label in fixed_labels or find(p.label) in fixed_labels):
            p.free = False
            p.value = fixed_labels.get(p.label, fixed_labels.get(find(p.label), p.value))
    groups: dict[str, int] = {}
    n_free = 0
    for p in params:
        if not p.free:
            continue
        if p.label:
            root = find(p.label)
            if root in groups:
                p.index = groups[root]
                continue
            groups[root] = n_free
        p.index = n_free
        n_free += 1
    # labels shared between a fixed and a free parameter
    for root in groups:
        if any(not p.free and p.label and find(p.label) == root for p in params):
            issues.append((0, f"Label '{root}' is used for both fixed and free parameters."))

    labels_all = {p.label for p in params if p.label}
    defined_names: list[str] = []
    for d in defined:
        unknown = [n for n in d.names if n not in labels_all and n not in defined_names]
        if unknown:
            issues.append((d.line, f"Unknown label '{unknown[0]}' in defined parameter "
                                   f"'{d.name}'. Label parameters first, e.g. 'y ~ a*x'."))
        if d.name in labels_all or d.name in order:
            issues.append((d.line, f"'{d.name}' is already used as a label or variable name."))
        defined_names.append(d.name)
    if issues:
        raise SEMSyntaxError(issues)

    warnings = []
    for f in latent:
        if len(indicators[f]) == 1:
            x = indicators[f][0]
            pv = by_key.get((x, "~~", x))
            if pv is not None and pv.free:
                warnings.append(f"Latent '{f}' has a single indicator; fix the residual variance "
                                f"of {x} (e.g. '{x} ~~ 0*{x}') to identify it.")
    return SEMModel(params, observed, latent, defined, indicators, endogenous, exogenous,
                    n_free, warnings, text, std_lv)


def validate(text: str, available: list[str] | None = None,
             numeric: set[str] | None = None) -> tuple[bool, str]:
    """(ok, message) for live feedback in the syntax editor."""
    try:
        model = build_model(text, available, numeric=numeric)
    except SEMSyntaxError as exc:
        return False, str(exc)
    p = len(model.observed)
    moments = p * (p + 1) // 2
    df = moments - model.n_free
    msg = (f"Syntax OK — {len(model.latent)} latent, {p} observed variables; "
           f"{model.n_free} free parameters, {moments} sample moments, df = {df}.")
    if df < 0:
        return False, msg + " The model is not identified (df < 0)."
    if model.warnings:
        msg += " " + " ".join(model.warnings)
    return True, msg


# ------------------------------------------------------------------ syntax generators
def cfa_syntax(constructs: list[tuple[str, list[str]]]) -> str:
    return "\n".join(f"{name} =~ {' + '.join(items)}" for name, items in constructs if items)


def sem_syntax(constructs: list[tuple[str, list[str]]], paths: list[tuple[str, str]],
               covariances: list[tuple[str, str]] = (), labels: bool = False) -> str:
    """Measurement part, then one regression line per outcome, then covariances. With
    ``labels`` paths get labels p1, p2, … so indirect effects can be defined."""
    lines = []
    meas = cfa_syntax(constructs)
    if meas:
        lines += ["# measurement model", meas]
    outcomes: dict[str, list[str]] = {}
    for src, dst in paths:
        outcomes.setdefault(dst, [])
        if src not in outcomes[dst]:
            outcomes[dst].append(src)
    if outcomes:
        lines.append("# structural model")
        k = 0
        for dst, srcs in outcomes.items():
            terms = []
            for s in srcs:
                k += 1
                terms.append(f"p{k}*{s}" if labels else s)
            lines.append(f"{dst} ~ {' + '.join(terms)}")
    if covariances:
        lines.append("# (residual) covariances")
        lines += [f"{a} ~~ {b}" for a, b in covariances]
    return "\n".join(lines)


def mediation_syntax(x: str, m: str, y: str) -> str:
    return (f"# simple mediation: {x} -> {m} -> {y}\n{m} ~ a*{x}\n{y} ~ c*{x} + b*{m}\n"
            f"indirect := a*b\ntotal := c + a*b")


def path_syntax(dependent: str, predictors: list[str]) -> str:
    return f"{dependent} ~ {' + '.join(predictors)}"

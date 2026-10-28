"""Origin's axis scale types as matplotlib scales, linked-axis formulas and tick formats.

Origin's Scale tab (docs.originlab.com/origin-help/axesdialog-scale-tab) defines:

* Probability ``X' = norminv(X/100)``; values in percent, range 0.0001–99.999.
* Probit: the same spacing, but a linear axis in probit units (5 = 50 %, one unit = one SD),
  so the tick labels are the probit values 3, 4, 5, 6, 7 …
* Reciprocal ``X' = 1/X`` and Offset Reciprocal ``X' = 1/(X + offset)`` (offset 273.15 for
  °C → K; Origin's page prints 273.14).
* Logit ``X' = ln(Y/(100 − Y))``; values in percent.
* Double Log Reciprocal (Weibull) ``X' = ln(−ln(1 − X))``; we take X in percent like the
  other probability scales, so ``X' = ln(−ln(1 − X/100))``.
* ln, log2, log10 (matplotlib's log scale) and symmetric log.

Each is registered as a matplotlib scale named ``rs_<type>`` with its own locator and
formatter. :class:`BrokenScale` wraps any base scale with one or more axis breaks, mapping the
kept segments to fractions of the axis length (Origin's break "position %"). No Qt here.
"""
from __future__ import annotations

import ast
import math

import numpy as np
from matplotlib import scale as mscale
from matplotlib import ticker
from matplotlib.transforms import Transform
from scipy import special

PCT_LO, PCT_HI = 1e-4, 100 - 1e-3     # Origin's probability range 0.0001–99.999 %

PROB_TICKS = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95,
              98, 99, 99.5, 99.9, 99.99, 99.999]
WEIBULL_TICKS = [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 63.2, 80, 90, 95, 99, 99.9, 99.99]


# ====================================================================== forward formulas
def probability(x):
    """Origin's Probability scale: norminv(X/100)."""
    p = np.clip(np.asarray(x, float), PCT_LO, PCT_HI) / 100.0
    return special.ndtri(p)


def probability_inv(t):
    return special.ndtr(np.asarray(t, float)) * 100.0


def probit(x):
    """Probit units: norminv(X/100) + 5."""
    return probability(x) + 5.0


def probit_inv(t):
    return probability_inv(np.asarray(t, float) - 5.0)


def logit(x):
    """Origin's Logit scale: ln(Y/(100 − Y))."""
    p = np.clip(np.asarray(x, float), PCT_LO, PCT_HI)
    return np.log(p / (100.0 - p))


def logit_inv(t):
    return 100.0 / (1.0 + np.exp(-np.asarray(t, float)))


def weibull(x):
    """Double log reciprocal: ln(−ln(1 − X/100))."""
    p = np.clip(np.asarray(x, float), PCT_LO, PCT_HI) / 100.0
    return np.log(-np.log1p(-p))


def weibull_inv(t):
    return -np.expm1(-np.exp(np.asarray(t, float))) * 100.0


def reciprocal(x, offset: float = 0.0):
    """1/(X + offset); offset 0 is Origin's Reciprocal scale."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.asarray(x, float) + offset
        v = np.where(v <= 0, 1e-300, v)
        return 1.0 / v


def reciprocal_inv(t, offset: float = 0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.asarray(t, float)
        t = np.where(t <= 0, 1e-300, t)
        return 1.0 / t - offset


# ====================================================================== matplotlib plumbing
class FuncTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, fwd, inv):
        super().__init__()
        self._fwd, self._inv = fwd, inv

    def transform_non_affine(self, values):
        return self._fwd(values)

    def inverted(self):
        return FuncTransform(self._inv, self._fwd)


class FixedInViewLocator(ticker.Locator):
    """Fixed candidate ticks, keeping those inside the view (at least two)."""

    def __init__(self, candidates):
        self.candidates = list(candidates)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        lo, hi = sorted((vmin, vmax))
        inside = [c for c in self.candidates if lo <= c <= hi]
        if len(inside) > 12:   # thin out a crowded probability axis
            keep = {0.01, 0.1, 1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99, 99.9, 99.99, 63.2}
            inside = [c for c in inside if c in keep] or inside
        return np.array(inside, float)


class ThinLocator(ticker.Locator):
    """Ticks evenly spaced in transformed space (reciprocal axes), mapped back to data and
    rounded to two significant figures."""

    def __init__(self, fwd, inv=None, n: int = 6):
        self.fwd, self.inv, self.n = fwd, inv, n

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        lo, hi = sorted((vmin, vmax))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return np.array([])
        t0, t1 = sorted(float(v) for v in self.fwd(np.array([lo, hi])))
        tt = ticker.MaxNLocator(nbins=self.n).tick_values(t0, t1)
        tt = tt[(tt > 0) & (tt >= t0) & (tt <= t1)]
        vals = []
        for v in self.inv(tt):
            if not np.isfinite(v) or v == 0:
                continue
            mag = 10 ** (math.floor(math.log10(abs(v))) - 1)
            r = round(v / mag) * mag
            if lo <= r <= hi:
                vals.append(float(f"{r:.6g}"))
        return np.array(sorted(set(vals)))


def _pct_fmt(x, _pos=None) -> str:
    if x >= 1 and x <= 99:
        return f"{x:g}"
    return f"{x:.6g}"


class _RSScale(mscale.ScaleBase):
    name = ""
    fwd = staticmethod(lambda v: v)
    inv = staticmethod(lambda v: v)
    domain = (-np.inf, np.inf)

    def __init__(self, *args, **kwargs):   # matplotlib < 3.11 passes the axis first
        super().__init__(args[0] if args else None)

    def get_transform(self):
        return FuncTransform(self.fwd, self.inv)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        lo, hi = self.domain
        if not np.isfinite(minpos):
            minpos = 1e-3
        vmin = vmin if lo < vmin < hi else (max(minpos, lo) if lo == 0 else lo + 1e-3)
        vmax = vmax if lo < vmax < hi else (min(hi - 1e-3, 99.99) if hi < np.inf
                                            else vmax)
        return vmin, vmax


class ProbabilityScale(_RSScale):
    name = "rs_probability"
    fwd, inv = staticmethod(probability), staticmethod(probability_inv)
    domain = (0.0, 100.0)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, PCT_LO) if vmin > 0 else 0.1, min(vmax, PCT_HI) if vmax < 100 \
            else 99.9

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedInViewLocator(PROB_TICKS))
        axis.set_major_formatter(ticker.FuncFormatter(_pct_fmt))
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())


class LogitScale(ProbabilityScale):
    name = "rs_logit"
    fwd, inv = staticmethod(logit), staticmethod(logit_inv)


class WeibullScale(ProbabilityScale):
    name = "rs_weibull"
    fwd, inv = staticmethod(weibull), staticmethod(weibull_inv)

    def set_default_locators_and_formatters(self, axis):
        super().set_default_locators_and_formatters(axis)
        axis.set_major_locator(FixedInViewLocator(WEIBULL_TICKS))


def probit_tick_positions(lo: float = PCT_LO, hi: float = PCT_HI) -> np.ndarray:
    """Percent values at whole probit units (1 … 9) inside [lo, hi]."""
    pos = probit_inv(np.arange(1.0, 10.0))
    return pos[(pos >= lo) & (pos <= hi)]


class ProbitScale(ProbabilityScale):
    name = "rs_probit"

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedInViewLocator(list(probit_tick_positions())))
        axis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _p=None: f"{float(probit(x)):.0f}"))
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())


class ReciprocalScale(_RSScale):
    name = "rs_reciprocal"

    def __init__(self, *args, offset: float = 0.0, **kwargs):
        super().__init__(args[0] if args else None)
        self.offset = float(offset)
        self.domain = (-self.offset, np.inf)

    def get_transform(self):
        off = self.offset
        return FuncTransform(lambda v: reciprocal(v, off), lambda t: reciprocal_inv(t, off))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        lo = -self.offset
        span = abs(vmax - vmin) or 1.0
        if vmin <= lo:
            vmin = minpos * 0.9 if (self.offset == 0 and np.isfinite(minpos) and minpos > 0) \
                else lo + max(span * 1e-3, 1e-9)
        if vmax <= lo:
            vmax = vmin + span
        return vmin, vmax

    def set_default_locators_and_formatters(self, axis):
        off = self.offset
        axis.set_major_locator(ThinLocator(lambda v: reciprocal(v, off),
                                           lambda t: reciprocal_inv(t, off)))
        axis.set_major_formatter(ticker.FuncFormatter(lambda v, _p=None: f"{v:g}"))
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())


class OffsetReciprocalScale(ReciprocalScale):
    name = "rs_offset_reciprocal"

    def __init__(self, *args, offset: float = 273.15, **kwargs):
        super().__init__(*args, offset=offset)


for _cls in (ProbabilityScale, ProbitScale, LogitScale, WeibullScale, ReciprocalScale,
             OffsetReciprocalScale):
    mscale.register_scale(_cls)

# axis.scale -> (matplotlib scale name, kwargs builder)
DOMAINS = {"log10": (0.0, np.inf), "log2": (0.0, np.inf), "ln": (0.0, np.inf),
           "probability": (0.0, 100.0), "probit": (0.0, 100.0), "logit": (0.0, 100.0),
           "weibull": (0.0, 100.0), "reciprocal": (0.0, np.inf)}
LABELS = {"linear": "Linear", "log10": "Log10", "log2": "Log2", "ln": "Ln",
          "probability": "Probability", "probit": "Probit", "logit": "Logit",
          "reciprocal": "Reciprocal", "offset_reciprocal": "Offset Reciprocal",
          "weibull": "Double Log Reciprocal (Weibull)", "symlog": "Symmetric Log"}


def mpl_scale(scale: str, offset: float = 273.15) -> tuple[str, dict]:
    """matplotlib scale name and kwargs for an Axis.scale value."""
    if scale == "log10":
        return "log", {"base": 10}
    if scale == "log2":
        return "log", {"base": 2}
    if scale == "ln":
        return "log", {"base": math.e}
    if scale == "symlog":
        return "symlog", {}
    if scale == "offset_reciprocal":
        return "rs_offset_reciprocal", {"offset": offset}
    if scale in ("probability", "probit", "logit", "weibull", "reciprocal"):
        return "rs_" + scale, {}
    return "linear", {}


def forward(scale: str, offset: float = 273.15):
    """Forward function X -> X' for an Axis.scale value (numpy, vectorised)."""
    return {"linear": lambda v: np.asarray(v, float), "log10": np.log10, "log2": np.log2,
            "ln": np.log, "probability": probability, "probit": probit, "logit": logit,
            "weibull": weibull, "reciprocal": reciprocal,
            "offset_reciprocal": lambda v: reciprocal(v, offset),
            "symlog": lambda v: np.sign(v) * np.log10(1 + np.abs(v))}[scale]


def domain_ok(scale: str, lo: float, hi: float, offset: float = 273.15) -> tuple[float, float]:
    """Clamp limits into the scale's domain (log/probability axes)."""
    if scale == "offset_reciprocal":
        dom = (-offset, np.inf)
    else:
        dom = DOMAINS.get(scale)
    if dom is None:
        return lo, hi
    a, b = dom
    if b == 100.0:
        lo = lo if lo > 0 else 0.1
        hi = hi if hi < 100 else 99.9
        if lo >= hi:
            lo, hi = 0.1, 99.9
        return lo, hi
    if lo <= a:
        lo = a + (min((hi - a) / 10, 1e-3) if hi > a else 1e-3)
    if hi <= lo:
        hi = lo * 10 if lo > 0 else lo + 1
    return lo, hi


# ====================================================================== axis breaks
class BrokenTransform(Transform):
    """Piecewise map of base-scale coordinates onto [0, 1]: kept segment k goes to
    [start_k, end_k]; break gaps are squeezed (linearly, so the map stays invertible)."""
    input_dims = output_dims = 1

    def __init__(self, base, knots_t, knots_u):
        super().__init__()
        self.base = base              # forward base function
        self.base_inv = None
        self.kt = np.asarray(knots_t, float)
        self.ku = np.asarray(knots_u, float)

    def _piece(self, t, kt, ku):
        t = np.asarray(t, float)
        out = np.interp(t, kt, ku)
        lo_slope = (ku[1] - ku[0]) / (kt[1] - kt[0])
        hi_slope = (ku[-1] - ku[-2]) / (kt[-1] - kt[-2])
        out = np.where(t < kt[0], ku[0] + (t - kt[0]) * lo_slope, out)
        out = np.where(t > kt[-1], ku[-1] + (t - kt[-1]) * hi_slope, out)
        return out

    def transform_non_affine(self, values):
        with np.errstate(all="ignore"):
            return self._piece(self.base(values), self.kt, self.ku)

    def inverted(self):
        return _BrokenInverse(self)


class _BrokenInverse(Transform):
    input_dims = output_dims = 1

    def __init__(self, fwd: BrokenTransform):
        super().__init__()
        self.fwd = fwd

    def transform_non_affine(self, values):
        t = self.fwd._piece(values, self.fwd.ku, self.fwd.kt)
        return self.fwd.base_inv(t)

    def inverted(self):
        return self.fwd


def break_layout(lo: float, hi: float, breaks: list, scale: str = "linear",
                 offset: float = 273.15, gap: float = 2.0):
    """Knots of a broken axis: (knots in base space, knots in axis fraction, gaps as
    [(u0, u1)] fractions). ``breaks`` = [{"from", "to", "position"}] in data units;
    position is the % of the axis length where the break sits (None = proportional)."""
    fwd = forward(scale, offset)
    lo, hi = sorted((lo, hi))
    brk = sorted([(min(b["from"], b["to"]), max(b["from"], b["to"]), b.get("position"))
                  for b in breaks if b.get("from") is not None and b.get("to") is not None],
                 key=lambda b: b[0])
    brk = [b for b in brk if lo < b[0] < hi and lo < b[1] < hi and b[1] > b[0]]
    t = lambda v: float(fwd(np.array([v]))[0])  # noqa: E731
    if not brk:
        return np.array([t(lo), t(hi)]), np.array([0.0, 1.0]), []
    g = max(0.0, min(20.0, gap)) / 100.0
    edges = [lo] + [v for b in brk for v in b[:2]] + [hi]
    seg_t = [(t(edges[2 * k]), t(edges[2 * k + 1])) for k in range(len(brk) + 1)]
    lengths = np.array([abs(b - a) for a, b in seg_t])
    usable = 1.0 - g * len(brk)
    auto = np.cumsum(lengths)[:-1] / lengths.sum() * usable + \
        (np.arange(len(brk)) + 0.5) * g
    centers = []
    for k, b in enumerate(brk):
        pos = b[2]
        centers.append(float(pos) / 100.0 if pos is not None else float(auto[k]))
    centers = np.clip(np.maximum.accumulate(np.array(centers)), g, 1 - g)
    kt, ku, gaps = [], [], []
    start = 0.0
    for k, (a, b) in enumerate(seg_t):
        end = centers[k] - g / 2 if k < len(brk) else 1.0
        kt += [a, b]
        ku += [start, end]
        if k < len(brk):
            gaps.append((end, centers[k] + g / 2))
            start = centers[k] + g / 2
    return np.array(kt), np.array(ku), gaps


class BrokenScale(mscale.ScaleBase):
    name = "rs_broken"

    def __init__(self, *args, lo=0.0, hi=1.0, breaks=(), base="linear", offset=273.15,
                 gap=2.0, **kwargs):
        super().__init__(args[0] if args else None)
        self.base, self.offset = base, offset
        self.kt, self.ku, self.gaps = break_layout(lo, hi, list(breaks), base, offset, gap)
        self.lo, self.hi = sorted((lo, hi))
        edges = [self.lo] + [v for b in sorted(breaks, key=lambda b: min(b["from"], b["to"]))
                             for v in sorted((b["from"], b["to"]))] + [self.hi]
        self.segments = [(edges[2 * k], edges[2 * k + 1]) for k in range(len(edges) // 2)]

    def get_transform(self):
        tr = BrokenTransform(forward(self.base, self.offset), self.kt, self.ku)
        tr.base_inv = _inverse(self.base, self.offset)
        return tr

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(SegmentLocator(self))
        if self.base in ("log10", "log2", "ln"):
            axis.set_major_formatter(ticker.LogFormatterSciNotation())
        else:
            axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())


def _inverse(scale: str, offset: float):
    return {"linear": lambda t: np.asarray(t, float), "log10": lambda t: 10 ** np.asarray(t),
            "log2": lambda t: 2 ** np.asarray(t), "ln": lambda t: np.exp(t),
            "probability": probability_inv, "probit": probit_inv, "logit": logit_inv,
            "weibull": weibull_inv, "reciprocal": reciprocal_inv,
            "offset_reciprocal": lambda t: reciprocal_inv(t, offset),
            "symlog": lambda t: np.sign(t) * (10 ** np.abs(t) - 1)}[scale]


class SegmentLocator(ticker.Locator):
    """Ticks of a broken axis: a base locator on every kept segment, more ticks on longer
    segments."""

    def __init__(self, broken: BrokenScale, step: float | None = None):
        self.b, self.step = broken, step

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        out = []
        n_seg = len(self.b.segments)
        for k, (a, c) in enumerate(self.b.segments):
            frac = self.b.ku[2 * k + 1] - self.b.ku[2 * k] if 2 * k + 1 < len(self.b.ku) \
                else 1.0 / n_seg
            nb = max(1, int(round(frac * 7)))
            if self.step:
                loc = ticker.MultipleLocator(self.step)
            elif self.b.base in ("log10", "log2", "ln"):
                loc = ticker.LogLocator(base={"log10": 10, "log2": 2}.get(self.b.base, math.e))
            else:
                loc = ticker.MaxNLocator(nbins=nb, steps=[1, 2, 2.5, 5, 10])
            ticks = np.asarray(loc.tick_values(a, c), float)
            span = c - a
            out += [v for v in ticks if a - span * 1e-9 <= v <= c + span * 1e-9]
        return np.array(sorted(set(out)))


mscale.register_scale(BrokenScale)


# ====================================================================== linked axis formulas
_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "ln": np.log,
          "log": np.log10, "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt,
          "abs": np.abs, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
          "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh}
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


class FormulaError(ValueError):
    pass


def compile_formula(text: str, var: str = "x"):
    """Safe vectorised function of ``var`` from Origin-like text ("1000/(x+273.15)",
    "1E7/x", "x^2"). Raises FormulaError."""
    src = (text or "").strip().lstrip("=").replace("^", "**")
    if not src:
        raise FormulaError("The formula is empty.")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise FormulaError(f"Syntax error in “{text}”.") from None
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise FormulaError(f"“{text}” uses something an axis formula cannot contain.")
        if isinstance(node, ast.Name) and node.id not in (var, "pi", "e") and \
                node.id not in _FUNCS:
            raise FormulaError(f"Unknown name “{node.id}” (use {var}).")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and
                                               node.func.id in _FUNCS):
            raise FormulaError("Only functions such as ln(), exp(), sqrt() can be called.")
    code = compile(tree, "<axis formula>", "eval")

    def fn(v):
        env = {var: np.asarray(v, float), "pi": math.pi, "e": math.e, **_FUNCS}
        with np.errstate(all="ignore"):
            out = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - AST-checked above
        return np.broadcast_to(np.asarray(out, float), np.shape(v)).astype(float)
    return fn


def link_functions(text: str, lo: float, hi: float):
    """(forward, inverse) for a top/right axis linked by ``text`` over the view [lo, hi].
    The inverse is numeric (monotone interpolation); non-monotone formulas are rejected."""
    f = compile_formula(text)
    lo, hi = sorted((lo, hi))
    span = hi - lo or 1.0
    grid = np.linspace(lo - 0.5 * span, hi + 0.5 * span, 4001)
    vals = f(grid)
    ok = np.isfinite(vals)
    # keep the connected finite part that covers the view
    if ok.sum() < 10:
        raise FormulaError(f"“{text}” is not defined over the axis range.")
    grid, vals = grid[ok], vals[ok]
    d = np.diff(vals)
    if not (np.all(d > 0) or np.all(d < 0)):
        inside = (grid >= lo) & (grid <= hi)
        g2, v2 = grid[inside], vals[inside]
        d2 = np.diff(v2)
        if not (np.all(d2 > 0) or np.all(d2 < 0)):
            raise FormulaError(f"“{text}” must be monotonic over the axis range.")
        grid, vals = g2, v2
    if vals[0] > vals[-1]:
        gi, vi = grid[::-1], vals[::-1]
    else:
        gi, vi = grid, vals

    def inverse(t):
        return np.interp(np.asarray(t, float), vi, gi)
    return f, inverse


class LinkedLocator(ticker.Locator):
    """Ticks of an axis linked by a formula: evenly spread along the parent axis, mapped
    through the formula and rounded to two significant figures (so a 1E7/x wavelength axis
    over a wavenumber axis gets readable, well-spread ticks)."""

    def __init__(self, fwd, parent_lim, n: int = 6):
        self.fwd, self.parent_lim, self.n = fwd, parent_lim, n

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        p0, p1 = sorted(self.parent_lim())
        lo, hi = sorted((vmin, vmax))
        pos = ticker.MaxNLocator(nbins=self.n).tick_values(p0, p1)
        vals = []
        for v in self.fwd(pos[(pos >= p0) & (pos <= p1)]):
            if not np.isfinite(v):
                continue
            if v == 0:
                vals.append(0.0)
                continue
            mag = 10 ** (math.floor(math.log10(abs(v))) - 1)
            r = float(f"{round(v / mag) * mag:.6g}")
            if lo - abs(hi - lo) * 1e-9 <= r <= hi + abs(hi - lo) * 1e-9:
                vals.append(r)
        return np.array(sorted(set(vals)))


# ====================================================================== tick formats
def _engineering(v: float) -> str:
    return ticker.EngFormatter(sep="")(v)


def tick_formatter(fmt: str, decimals: int = -1, prefix: str = "", suffix: str = "",
                   base=None):
    """Formatter for Origin's tick label Display types (Decimal, Scientific (1E3),
    Engineering (1k), ×10ⁿ, percent) with prefix/suffix."""
    dec = decimals if decimals >= 0 else None

    def wrap(text: str) -> str:
        return f"{prefix}{text}{suffix}" if text != "" else ""

    if fmt == "decimal":
        def f(v, _p=None):
            return wrap(f"{v:.{dec}f}" if dec is not None else f"{v:g}")
    elif fmt == "scientific":
        def f(v, _p=None):
            if v == 0:
                return wrap("0")
            return wrap(f"{v:.{dec if dec is not None else 1}E}".replace("E+0", "E")
                        .replace("E-0", "E-").replace("E+", "E"))
    elif fmt == "engineering":
        def f(v, _p=None):
            return wrap(_engineering(v))
    elif fmt == "power10":
        def f(v, _p=None):
            if v == 0:
                return wrap("0")
            exp = int(math.floor(math.log10(abs(v))))
            mant = v / 10 ** exp
            m = f"{mant:.{dec if dec is not None else 1}f}".rstrip("0").rstrip(".")
            if m in ("1", "-1"):
                return wrap(("-" if m == "-1" else "") + f"$10^{{{exp}}}$")
            return wrap(f"${m}\\times10^{{{exp}}}$")
    elif fmt == "percent":
        def f(v, _p=None):
            return wrap(f"{v * 100:.{dec if dec is not None else 0}f}%")
    else:
        if base is None:
            if not prefix and not suffix:
                return None
            inner = ticker.ScalarFormatter()
        else:
            inner = base

        def f(v, p=None):
            return wrap(inner(v, p) if inner is not None else f"{v:g}")
        formatter = ticker.FuncFormatter(f)
        formatter._inner = inner   # noqa: SLF001 - set_locs is forwarded below
        orig = formatter.set_locs

        def set_locs(locs, _orig=orig, _inner=inner):
            if hasattr(_inner, "set_locs"):
                _inner.axis = formatter.axis
                _inner.set_locs(locs)
            _orig(locs)
        formatter.set_locs = set_locs
        return formatter
    return ticker.FuncFormatter(f)

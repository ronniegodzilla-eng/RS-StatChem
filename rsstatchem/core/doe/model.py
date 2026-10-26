"""Response surface model fitting as in Design-Expert's analysis node.

Covers factor coding (numeric factors to -1/+1, categoric factors effect-coded, mixture
components to L-pseudo components), polynomial and Scheffé mixture model terms, least squares,
the Fit Summary (sequential model sums of squares, lack-of-fit tests, model summary statistics,
suggested model), the ANOVA with partial (Type III) sums of squares, lack of fit and pure
error, fit statistics (Std. Dev., Mean, C.V. %, R², Adjusted R², Predicted R², PRESS, Adeq
Precision), coefficient estimates in coded terms with SE, CI and VIF, final equations in coded
and actual factors, response transformations including Box-Cox, backward reduction respecting
hierarchy, 2-level factorial effects (half-normal / Pareto with Lenth's PSE) and case
diagnostics (externally studentized residuals, leverage, Cook's distance, DFFITS, DFBETAS).
Auto Select chooses terms forward, backward or stepwise by p-value (alpha in / out), AICc or
BIC. Split-plot designs (hard-to-change factors, whole plots in a Group column) are analysed
by REML with Kenward-Roger F tests and profile-likelihood variance intervals
(:class:`SplitPlotFit`); split-plot factorials split their effects into whole-plot and subplot
sets (:func:`split_plot_effects`).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.doe.designs import letter

SUP = {2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶"}


# ---------------------------------------------------------------------------- factors
@dataclass
class FactorDef:
    name: str
    column: str
    units: str = ""
    type: str = "numeric"  # numeric | categoric | mixture
    low: float = -1.0
    high: float = 1.0
    levels: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    index: int = 0
    hard: bool = False  # hard-to-change (whole-plot) factor of a split-plot design

    @property
    def letter(self) -> str:
        return letter(self.index)

    @property
    def categoric(self) -> bool:
        return self.type == "categoric"

    @property
    def center(self) -> float:
        return (self.low + self.high) / 2

    @property
    def half(self) -> float:
        return (self.high - self.low) / 2 or 1.0

    @property
    def title(self) -> str:
        return f"{self.letter}: {self.name}"

    @property
    def source(self) -> str:
        return f"{self.letter}-{self.name}"

    @property
    def axis_label(self) -> str:
        return f"{self.title}" + (f" ({self.units})" if self.units else "")

    def level_label(self, idx: int) -> str:
        return str(self.labels[idx]) if idx < len(self.labels) else str(self.levels[idx])

    def to_meta(self) -> dict:
        d = {"name": self.name, "column": self.column, "units": self.units, "type": self.type}
        if self.hard:
            d["changes"] = "hard"
        if self.categoric:
            d.update(levels=list(self.levels), labels=list(self.labels))
        else:
            d.update(low=self.low, high=self.high)
        return d


@dataclass
class Space:
    """The factors (and coding) a model is fitted in."""
    factors: list[FactorDef]
    kind: str = "process"
    total: float = 1.0
    block_col: str | None = None
    run_col: str | None = None
    std_col: str | None = None
    group_col: str | None = None  # whole plots of a split-plot design

    @property
    def split_plot(self) -> bool:
        return self.group_col is not None and any(f.hard for f in self.factors)

    @property
    def k(self) -> int:
        return len(self.factors)

    @property
    def mixture(self) -> bool:
        return self.kind == "mixture"

    @property
    def categoric(self) -> list[bool]:
        return [f.categoric for f in self.factors]

    @property
    def nlevels(self) -> list[int]:
        return [len(f.levels) if f.categoric else 0 for f in self.factors]

    @property
    def lower(self) -> np.ndarray:
        return np.array([f.low for f in self.factors], float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([f.high for f in self.factors], float)

    @property
    def pseudo_span(self) -> float:
        return self.total - float(self.lower.sum()) if self.mixture else 1.0

    def to_coded(self, actual: np.ndarray) -> np.ndarray:
        actual = np.atleast_2d(np.asarray(actual, float))
        out = np.empty_like(actual)
        for j, f in enumerate(self.factors):
            col = actual[:, j]
            if f.categoric:
                lv = np.asarray(f.levels, float)
                idx = np.array([int(np.argmin(np.abs(lv - v))) if np.isfinite(v) else -1
                                for v in col])
                out[:, j] = idx
            elif self.mixture:
                out[:, j] = (col - f.low) / self.pseudo_span
            else:
                out[:, j] = (col - f.center) / f.half
        return out

    def to_actual(self, coded: np.ndarray) -> np.ndarray:
        coded = np.atleast_2d(np.asarray(coded, float))
        out = np.empty_like(coded)
        for j, f in enumerate(self.factors):
            col = coded[:, j]
            if f.categoric:
                out[:, j] = np.asarray(f.levels, float)[col.astype(int)]
            elif self.mixture:
                out[:, j] = f.low + col * self.pseudo_span
            else:
                out[:, j] = f.center + col * f.half
        return out


def space_from_dataset(ds, factor_cols: list[str] | None = None,
                       overrides: dict | None = None, mixture: bool | None = None) -> Space:
    """Coding from ``ds.meta["doe"]``; factors missing there are coded from the data range."""
    from rsstatchem.core.stats.common import numeric

    meta = ds.meta.get("doe") or {}
    known = {f["column"].lower(): f for f in meta.get("factors", [])}
    if factor_cols is None:
        factor_cols = [f["column"] for f in meta.get("factors", []) if ds.has_name(f["column"])]
    kind = meta.get("kind", "process") if mixture is None else ("mixture" if mixture
                                                                else "process")
    overrides = overrides or {}
    factors = []
    for j, col in enumerate(factor_cols):
        var = ds.var(col)
        m = dict(known.get(var.name.lower(), {}))
        m.update(overrides.get(var.name, {}))
        x = numeric(ds, var.name).dropna().to_numpy(float)
        ftype = m.get("type") or ("mixture" if kind == "mixture" else "numeric")
        if kind == "mixture" and ftype == "numeric":
            ftype = "mixture"
        if ftype == "categoric":
            levels = m.get("levels") or sorted(set(x.tolist()))
            labels = m.get("labels") or [var.value_labels.get(float(v), _num(v)) for v in levels]
            f = FactorDef(m.get("name") or var.title, var.name, m.get("units", ""), "categoric",
                          levels=[float(v) for v in levels], labels=[str(s) for s in labels],
                          index=j, hard=m.get("changes") == "hard")
        else:
            low = m.get("low", float(x.min()) if len(x) else -1.0)
            high = m.get("high", float(x.max()) if len(x) else 1.0)
            if ftype == "mixture" and "low" not in m:
                low = 0.0
            f = FactorDef(m.get("name") or var.name, var.name, m.get("units", ""), ftype,
                          float(low), float(high), index=j, hard=m.get("changes") == "hard")
        factors.append(f)
    total = 1.0
    if kind == "mixture":
        total = float((meta.get("mixture") or {}).get("total", 0) or 0)
        if not total and factors:
            sums = sum(numeric(ds, f.column) for f in factors).dropna()
            total = float(np.round(sums.mean(), 10)) if len(sums) else 1.0
        if "mixture" not in meta:
            for f in factors:
                f.high = max(f.high, f.low)
    cols = meta.get("columns", {})
    block = cols.get("block") if cols.get("block") and ds.has_name(cols.get("block", "")) \
        else None
    if block is not None and numeric(ds, block).nunique() < 2:
        block = None
    run = cols.get("run") if ds.has_name(cols.get("run", "") or "") else None
    std = cols.get("std") if ds.has_name(cols.get("std", "") or "") else None
    group = cols.get("group") if ds.has_name(cols.get("group", "") or "") else None
    if group is not None and not any(f.hard for f in factors):
        group = None
    return Space(factors, kind, total, block, run, std, group)


def _num(v) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


@dataclass
class Observations:
    actual: np.ndarray
    coded: np.ndarray
    y: np.ndarray
    blocks: np.ndarray | None
    run: np.ndarray
    std: np.ndarray
    rows: np.ndarray
    z: np.ndarray | None = None  # response on the transformed scale (set by the fit)
    groups: np.ndarray | None = None  # whole-plot ids (split-plot designs)


def observations(ds, space: Space, response: str) -> Observations:
    from rsstatchem.core.stats.common import numeric

    frame = pd.DataFrame({f"f{j}": numeric(ds, f.column) for j, f in enumerate(space.factors)})
    frame["y"] = numeric(ds, response)
    if space.block_col:
        frame["block"] = numeric(ds, space.block_col)
    if space.group_col:
        frame["group"] = numeric(ds, space.group_col)
    frame["run"] = numeric(ds, space.run_col) if space.run_col else np.arange(1, ds.n_cases + 1)
    frame["std"] = numeric(ds, space.std_col) if space.std_col else np.arange(1, ds.n_cases + 1)
    frame = frame.dropna(subset=[c for c in frame.columns if c not in ("run", "std")])
    actual = frame[[f"f{j}" for j in range(space.k)]].to_numpy(float)
    coded = space.to_coded(actual) if len(actual) else np.zeros((0, space.k))
    for j, f in enumerate(space.factors):
        if f.categoric and np.any(coded[:, j] < 0):
            raise ValueError(f"Factor {f.name} has values that are not among its levels.")
    blocks = None
    if space.block_col:
        blocks = frame["block"].to_numpy(float)
        if len(np.unique(blocks)) < 2:
            blocks = None
    groups = frame["group"].to_numpy(float) if space.group_col else None
    return Observations(actual, coded, frame["y"].to_numpy(float), blocks,
                        frame["run"].to_numpy(float), frame["std"].to_numpy(float),
                        frame.index.to_numpy(), groups=groups)


# ---------------------------------------------------------------------------- terms
Term = tuple


def term_degree(t: Term) -> int:
    return 3 if t and t[0] == -1 else len(t)


def term_factors(t: Term) -> list[int]:
    return list(t[1:]) if t and t[0] == -1 else list(t)


def term_name(t: Term) -> str:
    if not t:
        return "Intercept"
    if t[0] == -1:
        a, b = letter(t[1]), letter(t[2])
        return f"{a}{b}({a}-{b})"
    counts: dict[int, int] = {}
    for i in t:
        counts[i] = counts.get(i, 0) + 1
    return "".join(letter(i) + SUP.get(c, "") if c > 1 else letter(i)
                   for i, c in sorted(counts.items()))


def term_key(t: Term) -> str:
    return ("d" + ".".join(map(str, t[1:]))) if t and t[0] == -1 else ".".join(map(str, t))


def key_term(key: str) -> Term:
    if key.startswith("d"):
        return (-1,) + tuple(int(s) for s in key[1:].split("."))
    return tuple(int(s) for s in key.split(".")) if key else ()


def sort_terms(terms) -> list[Term]:
    def key(t):
        base = sorted(set(term_factors(t)))
        return term_degree(t), t[0] == -1 if t else False, len(set(term_factors(t))) == 1 \
            and term_degree(t) > 1, base, t
    return sorted(set(terms), key=key)


def contains(small: Term, big: Term) -> bool:
    """Multiset containment of factor indices (A in AB, A in A², AB in A²B)."""
    if small == big or not small:
        return False
    if small[0] == -1 or big[0] == -1:
        return False
    rest = list(big)
    for i in small:
        if i in rest:
            rest.remove(i)
        else:
            return False
    return True


def process_terms(k: int, order: str, categoric: list[bool] | None = None) -> list[Term]:
    categoric = categoric or [False] * k
    num = [i for i in range(k) if not categoric[i]]
    terms: list[Term] = []
    if order == "mean":
        return []
    terms += [(i,) for i in range(k)]
    if order in ("2fi", "quadratic", "cubic", "3fi"):
        terms += list(itertools.combinations(range(k), 2))
    if order in ("quadratic", "cubic"):
        terms += [(i, i) for i in num]
    if order in ("cubic", "3fi"):
        terms += list(itertools.combinations(range(k), 3))
    if order == "cubic":
        for i in num:
            terms.append((i, i, i))
            for j in range(k):
                if j != i:
                    terms.append(tuple(sorted((i, i, j))))
    return sort_terms(terms)


def mixture_terms(q: int, order: str) -> list[Term]:
    if order == "mean":
        return []
    terms: list[Term] = [(i,) for i in range(q)]
    if order in ("quadratic", "special_cubic", "cubic"):
        terms += list(itertools.combinations(range(q), 2))
    if order in ("special_cubic", "cubic"):
        terms += list(itertools.combinations(range(q), 3))
    if order == "cubic":
        terms += [(-1, i, j) for i, j in itertools.combinations(range(q), 2)]
    return sort_terms(terms)


PROCESS_ORDERS = [("Mean", "mean"), ("Linear", "linear"), ("2FI", "2fi"),
                  ("Quadratic", "quadratic"), ("Cubic", "cubic")]
CATEGORIC_ORDERS = [("Mean", "mean"), ("Linear", "linear"), ("2FI", "2fi"), ("3FI", "3fi")]
MIXTURE_ORDERS = [("Mean", "mean"), ("Linear", "linear"), ("Quadratic", "quadratic"),
                  ("Special Cubic", "special_cubic"), ("Cubic", "cubic")]


def model_orders(space: Space) -> list[tuple[str, str]]:
    if space.mixture:
        return MIXTURE_ORDERS
    if all(space.categoric):
        return CATEGORIC_ORDERS
    return PROCESS_ORDERS


def model_terms(space: Space, order: str) -> list[Term]:
    if space.mixture:
        return mixture_terms(space.k, order)
    return process_terms(space.k, order, space.categoric)


def candidate_terms(space: Space) -> list[Term]:
    """Terms offered for manual model selection."""
    if space.mixture:
        return mixture_terms(space.k, "cubic")
    terms = set(process_terms(space.k, "cubic", space.categoric))
    for r in range(4, min(space.k, 6) + 1):
        terms |= set(itertools.combinations(range(space.k), r))
    return sort_terms(terms)


# ---------------------------------------------------------------------------- matrices
def _effect_codes(idx: np.ndarray, nlev: int, two_level_sign: bool = True) -> np.ndarray:
    """Effect (nominal) coding: level j -> +1 in column j, the last level -> -1 in every
    column (Design-Expert's "B[1], B[2]" terms). A two-level categoric factor gets one column
    coded -1 / +1 for its first / second level, like a numeric factor in a two-level
    factorial."""
    idx = idx.astype(int)
    if nlev == 2 and two_level_sign:
        return np.where(idx == 1, 1.0, -1.0)[:, None]
    E = np.zeros((len(idx), nlev - 1))
    for j in range(nlev - 1):
        E[:, j] = (idx == j).astype(float) - (idx == nlev - 1).astype(float)
    return E


def term_columns(coded: np.ndarray, t: Term, categoric=None, nlevels=None):
    n = len(coded)
    if t and t[0] == -1:
        a, b = coded[:, t[1]], coded[:, t[2]]
        return (a * b * (a - b))[:, None], [""]
    cols = np.ones((n, 1))
    suffixes = [""]
    counts: dict[int, int] = {}
    for i in t:
        counts[i] = counts.get(i, 0) + 1
    for i, c in sorted(counts.items()):
        if categoric is not None and categoric[i]:
            E = _effect_codes(coded[:, i], nlevels[i])
            cols = (cols[:, :, None] * E[:, None, :]).reshape(n, -1)
            suffixes = [s + (f"[{j + 1}]" if E.shape[1] > 1 else "") for s in suffixes
                        for j in range(E.shape[1])]
        else:
            cols = cols * (coded[:, i] ** c)[:, None]
    return cols, suffixes


def raw_matrix(coded: np.ndarray, terms: list[Term], mixture: bool = False,
               intercept: bool = True, categoric=None, nlevels=None) -> np.ndarray:
    parts = [np.ones((len(coded), 1))] if intercept else []
    for t in terms:
        parts.append(term_columns(coded, t, categoric, nlevels)[0])
    return np.hstack(parts) if parts else np.zeros((len(coded), 0))


def block_matrix(blocks: np.ndarray | None) -> tuple[np.ndarray, list]:
    if blocks is None:
        return np.zeros((0, 0)), []
    levels = sorted(set(blocks.tolist()))
    idx = np.array([levels.index(b) for b in blocks])
    return _effect_codes(idx, len(levels), two_level_sign=False), levels


@dataclass
class Matrix:
    X: np.ndarray
    groups: list  # (term, slice)
    labels: list[str]
    intercept: bool
    block_slice: slice | None
    block_levels: list


def build_matrix(space: Space, coded: np.ndarray, terms: list[Term],
                 blocks: np.ndarray | None = None) -> Matrix:
    intercept = not (space.mixture and any(len(t) == 1 for t in terms))
    parts, groups, labels = [], [], []
    col = 0
    if intercept:
        parts.append(np.ones((len(coded), 1)))
        groups.append(((), slice(0, 1)))
        labels.append("Intercept")
        col = 1
    B, levels = block_matrix(blocks)
    bslice = None
    if blocks is not None:
        parts.append(B)
        bslice = slice(col, col + B.shape[1])
        labels += [f"Block {levels[j]:g}" for j in range(B.shape[1])]
        col += B.shape[1]
    for t in terms:
        cols, suffixes = term_columns(coded, t, space.categoric, space.nlevels)
        parts.append(cols)
        groups.append((t, slice(col, col + cols.shape[1])))
        labels += [term_name(t) + s for s in suffixes]
        col += cols.shape[1]
    X = np.hstack(parts) if parts else np.zeros((len(coded), 0))
    return Matrix(X, groups, labels, intercept, bslice, levels)


def ratio(a: float, b: float) -> float:
    """a / b for F statistics; a perfect fit (b = 0) gives inf (or nan when a = 0 too)."""
    if b > 0:
        return a / b
    return float("inf") if a > 0 else float("nan")


def _ss_res(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    if X.shape[1] == 0:
        return float(y @ y), 0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r), int(np.linalg.matrix_rank(X))


def pure_error(obs: Observations) -> tuple[float, int]:
    keys = [tuple(np.round(row, 8)) + ((b,) if obs.blocks is not None else ())
            for row, b in zip(obs.coded, obs.blocks if obs.blocks is not None
                              else [None] * len(obs.y))]
    groups: dict = {}
    for key, i in zip(keys, range(len(keys))):
        groups.setdefault(key, []).append(i)
    ss, df = 0.0, 0
    for idx in groups.values():
        if len(idx) > 1:
            v = (obs.z if obs.z is not None else obs.y)[idx]
            ss += float(np.sum((v - v.mean()) ** 2))
            df += len(idx) - 1
    return ss, df


# ---------------------------------------------------------------------------- transforms
TRANSFORMS = [("none", "None"), ("sqrt", "Square Root"), ("ln", "Natural Log"),
              ("log10", "Base 10 Log"), ("inv_sqrt", "Inverse Sqrt"), ("inverse", "Inverse"),
              ("power", "Power"), ("boxcox", "Box-Cox (recommended lambda)")]


@dataclass
class Transform:
    kind: str = "none"
    k: float = 0.0
    lam: float = 1.0

    @property
    def power(self) -> float | None:
        return {"none": 1.0, "sqrt": 0.5, "ln": 0.0, "log10": 0.0, "inv_sqrt": -0.5,
                "inverse": -1.0}.get(self.kind, self.lam)

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float) + self.k
        if self.kind == "none":
            return y - self.k
        if self.kind == "log10":
            return np.log10(y)
        lam = self.power
        return np.log(y) if lam == 0 else y ** lam

    def inverse(self, z) -> np.ndarray:
        z = np.asarray(z, float)
        if self.kind == "none":
            return z
        if self.kind == "log10":
            return 10 ** z - self.k
        lam = self.power
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.exp(z) if lam == 0 else np.sign(z) * np.abs(z) ** (1 / lam)) - self.k

    def check(self, y: np.ndarray) -> str | None:
        if self.kind == "none":
            return None
        v = np.asarray(y, float) + self.k
        lam = self.power
        if (lam == 0 or lam < 0 or self.kind == "log10") and np.any(v <= 0):
            return "The transformation needs all response values + k > 0 (increase k)."
        if np.any(v < 0) and lam is not None and lam != int(lam):
            return "A fractional power needs all response values + k >= 0 (increase k)."
        return None

    def label(self, name: str) -> str:
        inner = f"{name} + {self.k:g}" if self.k else name
        if self.kind == "none":
            return name
        if self.kind == "log10":
            return f"Log10({inner})"
        lam = self.power
        if lam == 0:
            return f"Ln({inner})"
        if lam == 0.5:
            return f"Sqrt({inner})"
        if lam == -0.5:
            return f"1/Sqrt({inner})"
        if lam == -1:
            return f"1/({inner})"
        return f"({inner})^{lam:g}"

    def describe(self) -> str:
        if self.kind == "none":
            return "None"
        text = dict(TRANSFORMS).get(self.kind, self.kind)
        if self.kind in ("power", "boxcox"):
            text = f"Power (lambda = {self.lam:g})"
        return text + (f", k = {self.k:g}" if self.k else "")

    def to_meta(self) -> dict:
        return {"kind": self.kind, "k": self.k, "lam": self.lam}

    @classmethod
    def from_meta(cls, d: dict | None) -> "Transform":
        d = d or {}
        return cls(d.get("kind", "none"), float(d.get("k", 0.0)), float(d.get("lam", 1.0)))


# ---------------------------------------------------------------------------- the fit
class AliasedError(ValueError):
    pass


class Fit:
    """Least-squares fit of one response for a given set of terms."""

    def __init__(self, space: Space, obs: Observations, terms: list[Term],
                 transform: Transform | None = None, response: str = "R1",
                 response_name: str | None = None):
        self.space, self.obs = space, obs
        self.terms = sort_terms(terms)
        self.transform = transform or Transform()
        self.response = response
        self.response_name = response_name or response
        self.response_units = ""
        err = self.transform.check(obs.y)
        if err:
            raise ValueError(err)
        self.z = self.transform.apply(obs.y)
        obs.z = self.z
        self.mat = build_matrix(space, obs.coded, self.terms, obs.blocks)
        X = self.mat.X
        self.n, self.p = X.shape
        self.rank = int(np.linalg.matrix_rank(X)) if self.p else 0
        if self.rank < self.p:
            raise AliasedError("The model is aliased: some terms cannot be estimated from "
                               "this design. Choose a smaller model or remove terms.")
        if self.n <= self.p:
            raise ValueError(f"{self.n} observations are not enough for a model with "
                             f"{self.p} parameters.")
        self.XtX_inv = np.linalg.inv(X.T @ X)
        self.beta = self.XtX_inv @ X.T @ self.z
        self.fitted = X @ self.beta
        self.resid = self.z - self.fitted
        self.ss_res = float(self.resid @ self.resid)
        self.df_res = self.n - self.p
        self.ms_res = self.ss_res / self.df_res
        self.mean = float(self.z.mean())
        self.ss_cor = float(np.sum((self.z - self.mean) ** 2))
        self.h = np.einsum("ij,ij->i", X @ self.XtX_inv, X)
        self._anova = None

    # ------------------------------------------------------------------ helpers
    @property
    def n_blocks(self) -> int:
        return len(self.mat.block_levels)

    def _reduced_ss(self, drop: slice | list[slice], add_intercept: bool = False) -> float:
        drops = drop if isinstance(drop, list) else [drop]
        keep = np.ones(self.p, bool)
        for s in drops:
            keep[s] = False
        X = self.mat.X[:, keep]
        if add_intercept:
            X = np.hstack([np.ones((self.n, 1)), X])
        return _ss_res(X, self.z)[0]

    def _block_ss(self) -> tuple[float, float]:
        """(SS blocks, SS residual of intercept + blocks)."""
        ones = np.ones((self.n, 1))
        ss_mean = _ss_res(ones, self.z)[0]
        if self.mat.block_slice is None:
            return 0.0, ss_mean
        Xb = np.hstack([ones, self.mat.X[:, self.mat.block_slice]])
        ss_b = _ss_res(Xb, self.z)[0]
        return ss_mean - ss_b, ss_b

    # ------------------------------------------------------------------ ANOVA
    def anova(self) -> dict:
        if self._anova is not None:
            return self._anova
        ss_block, ss_after_block = self._block_ss()
        df_block = max(self.n_blocks - 1, 0)
        ss_model = ss_after_block - self.ss_res
        df_model = self.p - (1 if self.mat.intercept else 0) - df_block
        if not self.mat.intercept:
            df_model = self.p - df_block - 1
        rows = []
        lin = [s for t, s in self.mat.groups if len(t) == 1] if self.space.mixture else []
        if self.space.mixture and lin:
            ss = self._reduced_ss(lin, add_intercept=True) - self.ss_res
            rows.append(("Linear Mixture", ss, len(lin) - 1, ()))
        for t, s in self.mat.groups:
            if not t or (self.space.mixture and len(t) == 1):
                continue
            ss = self._reduced_ss(s) - self.ss_res
            name = self.space.factors[t[0]].source if len(t) == 1 else term_name(t)
            rows.append((name, ss, s.stop - s.start, t))
        ss_pe, df_pe = pure_error(self.obs)
        lof = None
        if df_pe > 0 and self.df_res - df_pe > 0:
            lof = (self.ss_res - ss_pe, self.df_res - df_pe, ss_pe, df_pe)
        self._anova = {"block": (ss_block, df_block) if df_block else None,
                       "model": (ss_model, df_model), "terms": rows,
                       "residual": (self.ss_res, self.df_res), "lof": lof,
                       "pure_error": (ss_pe, df_pe), "cor_total": (self.ss_cor, self.n - 1)}
        return self._anova

    def term_p_values(self) -> dict:
        out = {}
        for name, ss, df, t in self.anova()["terms"]:
            f = ratio(ss / df, self.ms_res) if df else float("nan")
            out[t] = (f, float(stats.f.sf(f, df, self.df_res)) if df else float("nan"))
        return out

    def model_test(self) -> tuple[float, float]:
        ss, df = self.anova()["model"]
        if df <= 0:
            return float("nan"), float("nan")
        f = ratio(ss / df, self.ms_res)
        return f, float(stats.f.sf(f, df, self.df_res))

    # ------------------------------------------------------------------ statistics
    def press(self) -> float | None:
        if np.any(self.h > 1 - 1e-10):
            return None
        return float(np.sum((self.resid / (1 - self.h)) ** 2))

    def fit_statistics(self) -> dict:
        a = self.anova()
        ss_model, df_model = a["model"]
        ss_tot = ss_model + self.ss_res
        df_tot = df_model + self.df_res
        sd = math.sqrt(self.ms_res)
        r2 = ss_model / ss_tot if ss_tot > 0 else float("nan")
        adj = 1 - (self.ss_res / self.df_res) / (ss_tot / df_tot) if ss_tot > 0 else float("nan")
        press = self.press()
        pred = 1 - press / ss_tot if press is not None and ss_tot > 0 else float("nan")
        prng = float(self.fitted.max() - self.fitted.min())
        adeq = prng / math.sqrt(self.p * self.ms_res / self.n) if self.ms_res > 0 else \
            float("nan")
        return {"std_dev": sd, "mean": self.mean,
                "cv": 100 * sd / abs(self.mean) if self.mean else float("nan"),
                "r2": r2, "adj_r2": adj, "pred_r2": pred, "press": press, "adeq": adeq}

    def coefficients(self, level: float = 0.95) -> list[dict]:
        se = np.sqrt(np.clip(np.diag(self.XtX_inv) * self.ms_res, 0, None))
        q = stats.t.ppf(0.5 + level / 2, self.df_res)
        vif = self.vif()
        rows = []
        bs = self.mat.block_slice
        for j, lab in enumerate(self.mat.labels):
            rows.append({"label": lab, "estimate": float(self.beta[j]), "se": float(se[j]),
                         "low": float(self.beta[j] - q * se[j]),
                         "high": float(self.beta[j] + q * se[j]), "vif": vif.get(j),
                         "block": bs is not None and bs.start <= j < bs.stop})
        if bs is not None:  # last block effect = -sum of the others (effect coding)
            c = np.zeros(self.p)
            c[bs] = -1
            est = float(c @ self.beta)
            s = math.sqrt(max(float(c @ self.XtX_inv @ c) * self.ms_res, 0))
            rows.insert(bs.stop, {"label": f"Block {self.mat.block_levels[-1]:g}",
                                  "estimate": est, "se": s, "low": est - q * s,
                                  "high": est + q * s, "vif": None, "block": True})
        return rows

    def vif(self) -> dict:
        X = self.mat.X
        bs = self.mat.block_slice
        cols = [j for j in range(self.p) if not (self.mat.intercept and j == 0)
                and not (bs is not None and bs.start <= j < bs.stop)]
        if not cols:
            return {}
        out = {}
        if self.space.mixture:
            inv = self.XtX_inv
            for j in cols:
                out[j] = float(inv[j, j] * (X[:, j] @ X[:, j]))
            return out
        Z = X[:, cols] - X[:, cols].mean(axis=0)
        norms = np.sqrt(np.sum(Z ** 2, axis=0))
        if np.any(norms == 0):
            return {}
        R = (Z / norms).T @ (Z / norms)
        try:
            d = np.diag(np.linalg.inv(R))
        except np.linalg.LinAlgError:
            return {}
        return {j: float(v) for j, v in zip(cols, d)}

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict:
        out = case_statistics(self.mat.X, self.resid, self.XtX_inv, self.ms_res, self.df_res)
        out.update({"run": self.obs.run, "std": self.obs.std, "actual": self.z,
                    "predicted": self.fitted, "residual": self.resid,
                    "labels": list(self.mat.labels)})
        return out

    def information(self) -> dict:
        """Likelihood statistics of the (normal, maximum-likelihood) fit for model selection:
        -2 log L = n ln(2 pi SSE / n) + n, AICc = -2 log L + 2k + 2k(k+1)/(n-k-1) and
        BIC = -2 log L + k ln n with k = coefficients + 1 (the residual variance)."""
        n = self.n
        k = self.p + 1
        m2ll = n * math.log(2 * math.pi * max(self.ss_res, 1e-300) / n) + n
        aicc = m2ll + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf)
        return {"m2ll": m2ll, "aic": m2ll + 2 * k, "aicc": aicc, "bic": m2ll + k * math.log(n),
                "k": k}

    # ------------------------------------------------------------------ prediction
    def row(self, coded: np.ndarray) -> np.ndarray:
        coded = np.atleast_2d(coded)
        parts = []
        if self.mat.intercept:
            parts.append(np.ones((len(coded), 1)))
        if self.mat.block_slice is not None:
            parts.append(np.zeros((len(coded), self.mat.block_slice.stop -
                                   self.mat.block_slice.start)))
        for t, _ in self.mat.groups:
            if t:
                parts.append(term_columns(coded, t, self.space.categoric,
                                          self.space.nlevels)[0])
        return np.hstack(parts)

    def predict_coded(self, coded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prediction on the transformed scale and the variance factor x'(X'X)^-1 x."""
        R = self.row(coded)
        return R @ self.beta, np.einsum("ij,ij->i", R @ self.XtX_inv, R)

    def predict(self, actual: np.ndarray) -> np.ndarray:
        """Back-transformed prediction (median on the original scale) at actual settings."""
        z, _ = self.predict_coded(self.space.to_coded(actual))
        return self.transform.inverse(z)

    def interval(self, actual: np.ndarray, level: float = 0.95) -> dict:
        z, v = self.predict_coded(self.space.to_coded(actual))
        q = stats.t.ppf(0.5 + level / 2, self.df_res)
        se_mean = np.sqrt(v * self.ms_res)
        se_pred = np.sqrt((1 + v) * self.ms_res)
        tr = self.transform

        def back(a, b):
            lo, hi = tr.inverse(a), tr.inverse(b)
            return np.minimum(lo, hi), np.maximum(lo, hi)

        median = tr.inverse(z)
        mean = median
        if tr.kind != "none":
            eps = 1e-4 * np.maximum(np.abs(z), 1)
            d2 = (tr.inverse(z + eps) - 2 * tr.inverse(z) + tr.inverse(z - eps)) / eps ** 2
            mean = median + 0.5 * d2 * self.ms_res
        ci = back(z - q * se_mean, z + q * se_mean)
        pi = back(z - q * se_pred, z + q * se_pred)
        return {"z": z, "median": median, "mean": mean, "se_mean": se_mean,
                "se_pred": se_pred, "ci": ci, "pi": pi, "sd": math.sqrt(self.ms_res)}

    # ------------------------------------------------------------------ equations
    def coded_equation(self) -> list[tuple[float, str]]:
        out = []
        for t, s in self.mat.groups:
            for j in range(s.start, s.stop):
                out.append((float(self.beta[j]), self.mat.labels[j] if t else ""))
        return out

    def actual_equations(self) -> list[tuple[str, list[tuple[float, str]]]]:
        """[(heading, [(coefficient, term)])] — one per categoric level combination."""
        if self.space.mixture:
            eq = mixture_actual_equation(self)
            return [("", eq)] if eq is not None else []
        cat = [j for j, f in enumerate(self.space.factors) if f.categoric]
        if len(cat) == self.space.k:
            return []
        combos = list(itertools.product(*[range(len(self.space.factors[j].levels))
                                          for j in cat])) or [()]
        out = []
        for combo in combos:
            point = np.ones(self.space.k)
            for j, lv in zip(cat, combo):
                point[j] = lv
            poly: dict[tuple, float] = {}
            if self.mat.intercept:
                poly[(0,) * self.space.k] = float(self.beta[0])
            for t, s in self.mat.groups:
                if not t:
                    continue
                cols, _ = term_columns(point[None, :], t, self.space.categoric,
                                       self.space.nlevels)
                coef = float(cols[0] @ self.beta[s])
                term_poly = {(0,) * self.space.k: coef}
                for i in t:
                    if self.space.factors[i].categoric:
                        continue
                    f = self.space.factors[i]
                    a, b = 1 / f.half, -f.center / f.half
                    new: dict[tuple, float] = {}
                    for e, c in term_poly.items():
                        e2 = list(e)
                        e2[i] += 1
                        new[tuple(e2)] = new.get(tuple(e2), 0.0) + c * a
                        new[e] = new.get(e, 0.0) + c * b
                    term_poly = new
                for e, c in term_poly.items():
                    poly[e] = poly.get(e, 0.0) + c
            eq = []
            for e in sorted(poly, key=lambda e: (sum(e), [-x for x in e])):
                eq.append((poly[e], _actual_label(self.space, e)))
            heading = ", ".join(f"{self.space.factors[j].name} = "
                                f"{self.space.factors[j].level_label(lv)}"
                                for j, lv in zip(cat, combo))
            out.append((heading, eq))
        return out


class SplitPlotFit(Fit):
    """REML fit of a split-plot design: y = X b + Z u + e with u ~ N(0, s2_wp I), one random
    effect per whole plot (Group), and e ~ N(0, s2 I) (Design-Expert's "ANOVA (REML)").

    The variance ratio eta = s2_wp / s2 maximises the restricted log-likelihood (s2 profiled
    out), b is the GLS estimate, and every fixed-effect test uses the Kenward-Roger (1997)
    small-sample adjusted covariance and denominator degrees of freedom, the method Stat-Ease
    documents for split-plot designs. For balanced designs this reproduces the classical
    split-plot ANOVA F tests exactly.
    """

    def __init__(self, space: Space, obs: Observations, terms: list[Term],
                 transform: Transform | None = None, response: str = "R1",
                 response_name: str | None = None):
        super().__init__(space, obs, terms, transform, response, response_name)
        if obs.groups is None:
            raise ValueError("A split-plot analysis needs the Group (whole-plot) column.")
        levels, gidx = np.unique(obs.groups, return_inverse=True)
        self.n_groups = len(levels)
        self.gidx = gidx
        Z = np.zeros((self.n, self.n_groups))
        Z[np.arange(self.n), gidx] = 1.0
        self.Z = Z
        self.m = Z.sum(axis=0)
        self.ols_ms_res = self.ms_res
        self._kr_cache: dict = {}
        self._reml()

    # ------------------------------------------------------------------ likelihood
    def _hinv(self, eta: float) -> np.ndarray:
        c = eta / (1 + eta * self.m)
        return np.eye(self.n) - (self.Z * c) @ self.Z.T

    def _profile(self, eta: float, ml: bool = False) -> tuple:
        X, z = self.mat.X, self.z
        Hi = self._hinv(eta)
        A = X.T @ Hi @ X
        beta = np.linalg.solve(A, X.T @ Hi @ z)
        r = z - X @ beta
        q = float(r @ Hi @ r)
        logdet_h = float(np.sum(np.log1p(eta * self.m)))
        n, p = self.n, self.p
        if ml:
            s2 = q / n
            ll = -0.5 * (n * math.log(max(s2, 1e-300)) + logdet_h) - 0.5 * n * (
                1 + math.log(2 * math.pi))
        else:
            s2 = q / (n - p)
            ll = -0.5 * ((n - p) * math.log(max(s2, 1e-300)) + logdet_h
                         + np.linalg.slogdet(A)[1])
        return ll, beta, s2, Hi, A

    def _maximise(self, ml: bool = False) -> float:
        def neg(u):
            return -self._profile(math.exp(u), ml)[0]

        grid = np.linspace(-14, 8, 45)
        vals = [neg(u) for u in grid]
        i = int(np.argmin(vals))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = optimize.minimize_scalar(neg, bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-10})
        best = min(res.fun, vals[i])
        eta = math.exp(res.x) if res.fun <= vals[i] else math.exp(grid[i])
        if -self._profile(0.0, ml)[0] <= best + 1e-12:  # boundary: no whole-plot variance
            eta = 0.0
        return eta

    def _reml(self) -> None:
        eta = self._maximise()
        ll, beta, s2, Hi, A = self._profile(eta)
        self.eta = eta
        self.s2 = s2
        self.s2_wp = eta * s2
        self.loglik_reml = ll
        X = self.mat.X
        self.Vinv = Hi / s2
        self.Phi = np.linalg.inv(A) * s2
        G = [self.Z @ self.Z.T, np.eye(self.n)]
        VX = self.Vinv @ X
        P = [VX.T @ g @ VX for g in G]
        Pm = self.Vinv - VX @ self.Phi @ VX.T
        info = np.array([[0.5 * np.trace(Pm @ G[i] @ Pm @ G[j]) for j in range(2)]
                         for i in range(2)])
        try:
            W = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(info)
        self.W, self.P = W, P
        Q = [[VX.T @ G[i] @ self.Vinv @ G[j] @ VX for j in range(2)] for i in range(2)]
        lam = np.zeros_like(self.Phi)
        for i in range(2):
            for j in range(2):
                lam += W[i, j] * (Q[i][j] - P[i] @ self.Phi @ P[j])
        self.Phi_A = self.Phi + 2 * self.Phi @ lam @ self.Phi
        self.beta = beta
        self.fitted = X @ beta
        self.resid = self.z - self.fitted
        self.ss_res = float(self.resid @ self.resid)
        total = self.s2 + self.s2_wp
        self.ms_res = total                      # variance of a new observation
        self.XtX_inv = self.Phi_A / total        # so x'(X'X)^-1 x * ms_res = var(x'b)
        # whitened regression (V = s2 L L') for the case diagnostics
        L = np.linalg.cholesky(np.eye(self.n) + eta * (self.Z @ self.Z.T))
        self._Xw = np.linalg.solve(L, X)
        self._ew = np.linalg.solve(L, self.z) - self._Xw @ beta
        self._XtXw_inv = np.linalg.inv(self._Xw.T @ self._Xw)
        self.h = np.einsum("ij,ij->i", self._Xw @ self._XtXw_inv, self._Xw)
        self._anova = None

    # ------------------------------------------------------------------ Kenward-Roger
    def kr_test(self, cols) -> tuple[float, float, float, float]:
        """Kenward-Roger F test of b[cols] = 0: (F, numerator df, denominator df, p)."""
        key = tuple(int(c) for c in cols)
        if key in self._kr_cache:
            return self._kr_cache[key]
        nl = len(key)
        L = np.zeros((self.p, nl))
        L[list(key), range(nl)] = 1.0
        Phi, W = self.Phi, self.W
        Theta = L @ np.linalg.inv(L.T @ Phi @ L) @ L.T
        M = [Theta @ Phi @ Pi @ Phi for Pi in self.P]
        A1 = sum(W[i, j] * np.trace(M[i]) * np.trace(M[j]) for i in range(2) for j in range(2))
        A2 = sum(W[i, j] * np.trace(M[i] @ M[j]) for i in range(2) for j in range(2))
        b = self.beta[list(key)]
        Fw = float(b @ np.linalg.solve(L.T @ self.Phi_A @ L, b)) / nl
        lc = float(nl)
        m, lam = 1e6, 1.0
        if A2 > 1e-14:  # as in pbkrtest (Halekoh & Hojsgaard 2014)
            B = (A1 + 6 * A2) / (2 * lc)
            g = ((lc + 1) * A1 - (lc + 4) * A2) / ((lc + 2) * A2)
            den = 3 * lc + 2 * (1 - g)
            c1, c2, c3 = g / den, (lc - g) / den, (lc + 2 - g) / den
            v0, v1, v2 = 1 + c1 * B, 1 - c2 * B, 1 - c3 * B
            v0 = 0.0 if abs(v0) < 1e-10 else v0
            if abs(v2) < 1e-10 or abs(v1) < 1e-14:
                m = 4.0
            else:
                rho = (1 / lc) * ((1 - A2 / lc) / v1) ** 2 * v0 / v2
                m = 4 + (lc + 2) / (lc * rho - 1) if abs(lc * rho - 1) > 1e-12 else 1e6
            if m <= 0:
                m = 1e6
            lam = 1.0 if abs(m - 2) < 1e-2 else m * (1 - A2 / lc) / (m - 2)
        F = lam * Fw
        p = float(stats.f.sf(F, nl, m))
        self._kr_cache[key] = (F, lc, m, p)
        return F, lc, m, p

    def is_whole_plot(self, t: Term) -> bool:
        return bool(t) and all(self.space.factors[i].hard for i in term_factors(t))

    def term_tests(self) -> list[tuple]:
        """[(term, source, F, df, error df, p)] in model order."""
        out = []
        for t, s in self.mat.groups:
            if not t:
                continue
            F, nl, m, p = self.kr_test(range(s.start, s.stop))
            name = self.space.factors[t[0]].source if len(t) == 1 else term_name(t)
            out.append((t, name, F, int(nl), m, p))
        return out

    def term_p_values(self) -> dict:
        return {t: (F, p) for t, _n, F, _l, _m, p in self.term_tests()}

    def section_test(self, whole: bool) -> tuple | None:
        cols = [c for t, s in self.mat.groups if t and self.is_whole_plot(t) == whole
                for c in range(s.start, s.stop)]
        return self.kr_test(cols) if cols else None

    def model_test(self) -> tuple[float, float]:
        cols = [c for t, s in self.mat.groups if t for c in range(s.start, s.stop)]
        if not cols:
            return float("nan"), float("nan")
        F, _l, _m, p = self.kr_test(cols)
        return F, p

    def anova(self) -> dict:
        if self._anova is None:
            tests = self.term_tests()
            self._anova = {"kind": "reml",
                           "whole": [r for r in tests if self.is_whole_plot(r[0])],
                           "sub": [r for r in tests if not self.is_whole_plot(r[0])],
                           "whole_test": self.section_test(True),
                           "sub_test": self.section_test(False),
                           "block": (self.n_blocks - 1) if self.n_blocks else 0}
        return self._anova

    # ------------------------------------------------------------------ statistics
    def press(self) -> float | None:
        return None

    def information(self) -> dict:
        """Maximum-likelihood statistics (the fixed effects differ between candidate models, so
        model selection uses ML rather than REML); k = coefficients + 2 variances."""
        eta = self._maximise(ml=True)
        ll = self._profile(eta, ml=True)[0]
        n, k = self.n, self.p + 2
        m2ll = -2 * ll
        aicc = m2ll + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf)
        return {"m2ll": m2ll, "aic": m2ll + 2 * k, "aicc": aicc, "bic": m2ll + k * math.log(n),
                "k": k, "m2ll_reml": -2 * self.loglik_reml}

    def fit_statistics(self) -> dict:
        sd = math.sqrt(self.s2)
        r2 = 1 - self.ss_res / self.ss_cor if self.ss_cor > 0 else float("nan")
        adj = 1 - (1 - r2) * (self.n - 1) / (self.n - self.p) if self.n > self.p else \
            float("nan")
        prng = float(self.fitted.max() - self.fitted.min())
        avg_var = float(np.mean(np.einsum("ij,ij->i", self.mat.X @ self.Phi_A, self.mat.X)))
        return {"std_dev": sd, "mean": self.mean,
                "cv": 100 * sd / abs(self.mean) if self.mean else float("nan"),
                "r2": r2, "adj_r2": adj, "pred_r2": float("nan"), "press": None,
                "adeq": prng / math.sqrt(avg_var) if avg_var > 0 else float("nan")}

    # ------------------------------------------------------------------ variance intervals
    def reml_loglik(self, s2: float, s2_wp: float) -> float:
        """Restricted log-likelihood (additive constant dropped) at residual variance ``s2``
        and group variance ``s2_wp``, from the n x p and groups x p cross products only
        (V = s2 (I + eta Z Z'), eta = s2_wp / s2; Patterson & Thompson 1971)."""
        if not hasattr(self, "_xp"):
            X, z, Z = self.mat.X, self.z, self.Z
            self._xp = (X.T @ X, Z.T @ X, X.T @ z, Z.T @ z, float(z @ z))
        XtX, ZtX, Xtz, Ztz, ztz = self._xp
        eta = s2_wp / s2
        c = eta / (1 + eta * self.m)
        A = XtX - ZtX.T @ (c[:, None] * ZtX)
        b = Xtz - ZtX.T @ (c * Ztz)
        sign, ld_a = np.linalg.slogdet(A)
        q = ztz - float(np.sum(c * Ztz ** 2)) - float(b @ np.linalg.solve(A, b))
        return -0.5 * ((self.n - self.p) * math.log(s2) + float(np.sum(np.log1p(eta * self.m)))
                       + ld_a + q / s2)

    def _profile_loglik(self, which: str, value: float) -> float:
        """Profile restricted log-likelihood with one variance fixed at ``value`` and the other
        maximised (bounded search on the log scale; the group variance may be 0)."""
        if which == "Group":
            def neg(u):
                return -self.reml_loglik(math.exp(u), value)

            c0 = math.log(self.s2)
            res = optimize.minimize_scalar(neg, bounds=(c0 - 12, c0 + 12), method="bounded",
                                           options={"xatol": 1e-9})
            return -float(res.fun)
        s2 = max(value, 1e-300)
        at_zero = self.reml_loglik(s2, 0.0)

        def neg(u):
            return -self.reml_loglik(s2, math.exp(u))

        c0 = math.log(max(self.s2_wp, 1e-3 * self.s2, 1e-300))
        res = optimize.minimize_scalar(neg, bounds=(c0 - 20, c0 + 12), method="bounded",
                                       options={"xatol": 1e-9})
        return max(-float(res.fun), at_zero)

    def profile_interval(self, which: str, level: float = 0.95) -> tuple[float, float]:
        """Profile-likelihood interval of the ``Group`` (whole-plot) or ``Residual``
        variance: the values whose restricted likelihood-ratio statistic
        2 (l_max - l_profile) stays below the chi-square(1) quantile (3.84 for 95 %). The
        group variance is bounded below by 0 (lower limit 0 when 0 is inside the region)."""
        est = self.s2_wp if which == "Group" else self.s2
        crit = float(stats.chi2.ppf(level, 1))
        l_max = self.reml_loglik(self.s2, self.s2_wp)

        def g(v):
            return 2 * (l_max - self._profile_loglik(which, v)) - crit

        if est > 0 and g(0.0 if which == "Group" else est * 1e-12) > 0:
            low = optimize.brentq(g, 0.0 if which == "Group" else est * 1e-12, est,
                                  xtol=1e-12 * max(est, 1e-300), rtol=1e-10)
        else:
            low = 0.0
        below, hi = est, max(est, 0.05 * (self.s2 + self.s2_wp), 1e-12)
        for _ in range(200):
            hi *= 2.0
            if g(hi) > 0:
                break
            below = hi
        else:
            return float(low), math.inf
        high = optimize.brentq(g, below, hi, xtol=1e-12 * hi, rtol=1e-10)
        return float(low), float(high)

    def variance_components(self, level: float = 0.95) -> list[dict]:
        """Group (whole-plot) and residual variances with standard errors from the inverse
        REML information and profile-likelihood intervals (:meth:`profile_interval`), which
        are never negative and, unlike Wald intervals, follow the skewness of the
        likelihood."""
        out = []
        total = self.s2 + self.s2_wp
        for name, v, i in (("Group", self.s2_wp, 0), ("Residual", self.s2, 1)):
            se = math.sqrt(max(self.W[i, i], 0.0))
            try:
                low, high = self.profile_interval(name, level)
            except (ValueError, np.linalg.LinAlgError, ArithmeticError):
                q = stats.norm.ppf(0.5 + level / 2)  # fall back to a log-scale Wald interval
                f = math.exp(q * se / v) if v > 0 else math.nan
                low, high = (v / f, v * f) if v > 0 else (0.0, q * se)
            out.append({"source": name, "variance": v, "se": se, "low": low,
                        "high": high, "percent": 100 * v / total if total else 0.0})
        return out

    def coefficients(self, level: float = 0.95) -> list[dict]:
        rows = super().coefficients(level)
        se = np.sqrt(np.clip(np.diag(self.Phi_A), 0, None))
        for j, lab in enumerate(self.mat.labels):
            r = next((r for r in rows if r["label"] == lab), None)
            if r is None:
                continue
            _F, _l, m, _p = self.kr_test([j])
            q = stats.t.ppf(0.5 + level / 2, m)
            r.update(se=float(se[j]), low=float(self.beta[j] - q * se[j]),
                     high=float(self.beta[j] + q * se[j]), df=m,
                     whole_plot=self._column_is_whole(j))
        return rows

    def _column_is_whole(self, j: int) -> bool:
        for t, s in self.mat.groups:
            if s.start <= j < s.stop:
                return self.is_whole_plot(t) if t else True
        return True  # blocks

    def diagnostics(self) -> dict:
        out = case_statistics(self._Xw, self._ew, self._XtXw_inv, self.s2, self.df_res)
        out.update({"run": self.obs.run, "std": self.obs.std, "actual": self.z,
                    "predicted": self.fitted, "residual": self.resid,
                    "labels": list(self.mat.labels)})
        return out


def case_statistics(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, s2: float,
                    df: int) -> dict:
    """Leverage, studentized residuals, Cook's distance, DFFITS and DFBETAS of a least-squares
    fit (Belsley, Kuh & Welsch). ``DFBETAS[i, j]`` is the change of coefficient j when run i
    is left out, in units of its standard error estimated without run i."""
    n, p = X.shape
    h = np.einsum("ij,ij->i", X @ XtX_inv, X)
    hc = np.clip(h, 0, 1 - 1e-12)
    s = math.sqrt(max(s2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = resid / (s * np.sqrt(1 - hc))
        t = r * np.sqrt((df - 1) / np.clip(df - r ** 2, 1e-12, None)) if df > 1 else r * 0
        cook = r ** 2 * hc / (p * (1 - hc))
        dffits = t * np.sqrt(hc / (1 - hc))
        s_i2 = np.clip((df * s2 - resid ** 2 / (1 - hc)) / max(df - 1, 1), 1e-300, None)
        C = XtX_inv @ X.T                                   # p x n
        dfbetas = (C * (resid / (1 - hc))).T / np.sqrt(np.outer(s_i2, np.diag(XtX_inv)))
    return {"leverage": h, "int_stud": r, "ext_stud": t, "cooks": cook, "dffits": dffits,
            "dfbetas": dfbetas}


def _actual_label(space: Space, exps: tuple) -> str:
    parts = []
    for i, e in enumerate(exps):
        if e:
            parts.append(space.factors[i].name + (SUP.get(e, f"^{e}") if e > 1 else ""))
    return " * ".join(parts)


def mixture_actual_equation(fit: Fit) -> list[tuple[float, str]] | None:
    """Scheffé equation in actual component amounts (exact refit on the simplex plane)."""
    sp = fit.space
    q = sp.k
    rng = np.random.default_rng(0)
    pts = rng.dirichlet(np.ones(q), size=max(200, 20 * fit.p)) * sp.total
    z, _ = fit.predict_coded(sp.to_coded(pts))
    terms = [t for t, _ in fit.mat.groups if t]
    if not fit.mat.intercept:
        A = raw_matrix(pts, terms, mixture=True, intercept=False)
    else:
        return None
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    if np.max(np.abs(A @ coef - z)) > 1e-6 * max(1.0, np.max(np.abs(z))):
        return None
    out = []
    for c, t in zip(coef, terms):
        if t[0] == -1:
            a, b = sp.factors[t[1]].name, sp.factors[t[2]].name
            label = f"{a} * {b} * ({a} - {b})"
        else:
            label = " * ".join(sp.factors[i].name for i in t)
        out.append((float(c), label))
    return out


# ---------------------------------------------------------------------------- model building
def drop_aliased(space: Space, obs: Observations, terms: list[Term]) -> tuple[list, list]:
    """Terms that can be estimated (in model order) and those aliased with earlier ones."""
    ordered = sort_terms(terms)
    X = build_matrix(space, obs.coded, ordered, obs.blocks).X
    if X.shape[1] <= X.shape[0] and np.linalg.matrix_rank(X) == X.shape[1]:
        return ordered, []  # full rank: nothing aliased (and no term-by-term checks)
    kept, dropped = [], []
    for t in ordered:
        mat = build_matrix(space, obs.coded, kept + [t], obs.blocks)
        if np.linalg.matrix_rank(mat.X) < mat.X.shape[1]:
            dropped.append(t)
        else:
            kept.append(t)
    return kept, dropped


def removable(t: Term, terms: list[Term], space: Space, hierarchy: bool) -> bool:
    if not t:
        return False
    if space.mixture and len(t) == 1:
        return False
    if hierarchy and not space.mixture:
        return not any(contains(t, u) for u in terms if u != t)
    return True


def fit_class(space: Space):
    """Least squares, or REML for a split-plot design (whole plots as a random effect)."""
    return SplitPlotFit if space.split_plot else Fit


def missing_parents(terms: list[Term], candidates: list[Term], space: Space) -> list[Term]:
    """Lower-order terms that a hierarchical model needs but ``terms`` lacks."""
    if space.mixture:
        return []
    return sort_terms({u for t in terms for u in candidates if contains(u, t)
                       and u not in terms})


SELECTION_CRITERIA = [("p", "p-value"), ("aicc", "AICc"), ("bic", "BIC")]
SELECTION_METHODS = [("backward", "Backward"), ("forward", "Forward"), ("stepwise", "Stepwise")]


def auto_select(space: Space, obs: Observations, terms: list[Term], transform: Transform,
                method: str = "backward", criterion: str = "p", alpha_in: float = 0.10,
                alpha_out: float = 0.10, hierarchy: bool = True, response: str = "R1",
                forced: list[Term] | None = None):
    """Design-Expert's Auto Select. ``terms`` are the candidates (the model chosen on the
    Model tab); blocks, forced terms and (mixtures) the linear blending terms always stay in.

    * backward: start with all candidates; remove the term with the largest p-value while it
      exceeds alpha out (criterion p), or the term whose removal lowers AICc / BIC most.
    * forward: start with the forced terms; add the term with the smallest p-value while it is
      below alpha in, or the term that lowers AICc / BIC most. Every candidate may enter (as
      in Design-Expert); with hierarchy on, the parents a selected term needs are added at the
      end (steps marked "hierarchy").
    * stepwise: forward steps, each followed by backward removals (alpha out).

    Returns (fit, steps) with steps [(action, term, value)] (value = p or the criterion).
    """
    cls = fit_class(space)
    candidates = sort_terms(terms)
    keep = set(forced or [])
    if space.mixture:
        keep |= {t for t in candidates if len(t) == 1}

    def make(ts):
        return cls(space, obs, sort_terms(ts), transform, response)

    def score(fit):
        return fit.information()[criterion]

    def p_of(fit, t):
        return fit.term_p_values().get(t, (math.nan, math.nan))[1]

    steps: list = []
    if method == "backward":
        current = list(candidates)
    else:
        current = [t for t in candidates if t in keep]
    fit = make(current)
    seen = {frozenset(current)}

    def backward_step(fit, current, alpha) -> tuple:
        cands = [t for t in current if t not in keep and removable(t, current, space,
                                                                    hierarchy)]
        if not cands:
            return None
        if criterion == "p":
            pv = fit.term_p_values()
            scored = [(pv.get(t, (0, math.nan))[1], t) for t in cands]
            scored = [c for c in scored if np.isfinite(c[0])]
            if not scored:
                return None
            p, worst = max(scored, key=lambda c: (c[0], term_degree(c[1])))
            if p <= alpha:
                return None
            trial = [u for u in current if u != worst]
            return make(trial), trial, ("remove", worst, p)
        best = None
        base = score(fit)
        for t in cands:
            trial = [u for u in current if u != t]
            try:
                f = make(trial)
            except (ValueError, np.linalg.LinAlgError):
                continue
            v = score(f)
            if v < base - 1e-9 and (best is None or v < best[0]):
                best = (v, f, trial, t)
        if best is None:
            return None
        return best[1], best[2], ("remove", best[3], best[0])

    def forward_step(fit, current) -> tuple:
        cands = [t for t in candidates if t not in current]
        best = None
        base = score(fit) if criterion != "p" else None
        for t in cands:
            trial = current + [t]
            if frozenset(trial) in seen:
                continue
            try:
                f = make(trial)
            except (ValueError, np.linalg.LinAlgError):
                continue
            v = p_of(f, t) if criterion == "p" else score(f)
            if not np.isfinite(v):
                continue
            if best is None or v < best[0]:
                best = (v, f, trial, t)
        if best is None:
            return None
        if criterion == "p" and best[0] >= alpha_in:
            return None
        if criterion != "p" and not best[0] < base - 1e-9:
            return None
        return best[1], best[2], ("add", best[3], best[0])

    for _ in range(4 * len(candidates) + 4):
        if method == "backward":
            res = backward_step(fit, current, alpha_out)
            if res is None:
                break
            fit, current, step = res
            steps.append(step)
            continue
        res = forward_step(fit, current)
        if res is None:
            break
        fit, current, step = res
        steps.append(step)
        seen.add(frozenset(current))
        if method == "stepwise" and len([t for t in current if t not in keep]) >= 2:
            while True:
                back = backward_step(fit, current, alpha_out)
                if back is None or back[2][1] == step[1]:
                    break
                fit, current, bstep = back
                steps.append(bstep)
                seen.add(frozenset(current))
    if hierarchy and method != "backward":
        parents = missing_parents(current, candidates, space)
        if parents:
            current = current + parents
            fit = make(current)
            steps += [("hierarchy", t, float("nan")) for t in parents]
    return fit, steps


def backward(space: Space, obs: Observations, terms: list[Term], transform: Transform,
             alpha: float = 0.1, hierarchy: bool = True, response: str = "R1"):
    """Backward elimination by p-value (alpha out), keeping the model hierarchical.
    Returns (fit, [(term, p)])."""
    fit, steps = auto_select(space, obs, terms, transform, "backward", "p", alpha_out=alpha,
                             hierarchy=hierarchy, response=response)
    return fit, [(t, v) for _, t, v in steps]


@dataclass
class FitSummary:
    sequential: list  # (source, ss, df, ms, f, p, remark)
    lof: list
    summary: list     # (source, sd, r2, adj, pred, press, remark)
    suggested: str
    suggested_order: str
    pure_error: tuple
    aliased: dict


def fit_summary(space: Space, obs: Observations, transform: Transform | None = None
                ) -> FitSummary:
    transform = transform or Transform()
    err = transform.check(obs.y)
    if err:
        raise ValueError(err)
    z = transform.apply(obs.y)
    obs.z = z
    n = len(z)
    B, _ = block_matrix(obs.blocks)
    has_blocks = obs.blocks is not None
    ones = np.ones((n, 1))
    ss_pe, df_pe = pure_error(obs)
    seq = [("Mean vs Total", float(n * z.mean() ** 2), 1, None, None)]
    prev_ss, prev_rank = _ss_res(ones, z)
    prev_name = "Mean"
    ss_tot_cor = prev_ss
    if has_blocks:
        ss_b, r_b = _ss_res(np.hstack([ones, B]), z)
        seq.append(("Block vs Mean", prev_ss - ss_b, r_b - prev_rank, None, None))
        prev_ss, prev_rank, prev_name = ss_b, r_b, "Block"
    ss_after_block = prev_ss
    lof, summary, aliased = [], [], {}
    seq_rows, last_res = [], (prev_ss, n - prev_rank)
    suggested, suggested_order = "Mean", "mean"
    terms_prev: list = []
    for name, order in model_orders(space):
        if order == "mean":
            continue
        terms = model_terms(space, order)
        if not terms or set(terms) == set(terms_prev):
            continue
        new = len(set(terms) - set(terms_prev))
        mat = build_matrix(space, obs.coded, terms, obs.blocks)
        ss, rank = _ss_res(mat.X, z)
        df_res = n - rank
        is_aliased = rank < mat.X.shape[1]
        aliased[order] = is_aliased
        d_df = rank - prev_rank
        d_ss = prev_ss - ss
        f = p = None
        if d_df > 0 and df_res > 0 and ss > 1e-14 * max(ss_tot_cor, 1):
            f = (d_ss / d_df) / (ss / df_res)
            p = float(stats.f.sf(f, d_df, df_res))
        remark = "Aliased" if is_aliased or d_df < new else ""
        seq_rows.append([f"{name} vs {prev_name}", d_ss, d_df, f, p, remark, order])
        if df_pe > 0 and df_res - df_pe > 0:
            fl = ratio((ss - ss_pe) / (df_res - df_pe), ss_pe / df_pe)
            lof.append([name, ss - ss_pe, df_res - df_pe, fl, float(stats.f.sf(
                fl, df_res - df_pe, df_pe)), remark, order])
        else:
            lof.append([name, ss - ss_pe if df_res - df_pe > 0 else 0.0,
                        max(df_res - df_pe, 0), None, None, remark, order])
        ss_model = ss_after_block - ss
        tot = ss_model + ss
        df_model = rank - (n - len(z)) - (r_b if has_blocks else 1)
        sd = math.sqrt(ss / df_res) if df_res > 0 else float("nan")
        r2 = ss_model / tot if tot > 0 else float("nan")
        adj = 1 - (ss / df_res) / (tot / (df_model + df_res)) if df_res > 0 and tot > 0 \
            else float("nan")
        press = None
        if not is_aliased and df_res > 0:
            H = mat.X @ np.linalg.pinv(mat.X)
            h = np.diag(H)
            if np.all(h < 1 - 1e-10):
                e = z - H @ z
                press = float(np.sum((e / (1 - h)) ** 2))
        pred = 1 - press / tot if press is not None and tot > 0 else float("nan")
        summary.append([name, sd, r2, adj, pred, press, remark, order])
        if not remark and p is not None and p < 0.05:
            suggested, suggested_order = name, order
        prev_ss, prev_rank, prev_name, terms_prev = ss, rank, name, terms
        last_res = (ss, df_res)
    for rows in (seq_rows, lof, summary):
        for r in rows:
            if r[-1] == suggested_order and r[-2] == "":
                r[-2] = "Suggested"
    sequential = [list(s) + [""] for s in seq] + [r[:6] for r in seq_rows]
    sequential.append(["Residual", last_res[0], last_res[1], None, None, ""])
    sequential.append(["Total", float(z @ z), n, None, None, ""])
    return FitSummary(sequential, [r[:6] for r in lof], [r[:7] for r in summary],
                      suggested, suggested_order, (ss_pe, df_pe), aliased)


# ---------------------------------------------------------------------------- Box-Cox
def box_cox(space: Space, obs: Observations, terms: list[Term], k: float = 0.0,
            level: float = 0.95, current: float = 1.0) -> dict:
    """ln(Residual SS) of the geometric-mean scaled power transform across lambda."""
    y = obs.y + k
    if np.any(y <= 0):
        raise ValueError("Box-Cox needs all response values + k > 0.")
    X = build_matrix(space, obs.coded, terms, obs.blocks).X
    gm = math.exp(float(np.mean(np.log(y))))

    def ss(lam):
        if abs(lam) < 1e-9:
            w = gm * np.log(y)
        else:
            w = (y ** lam - 1) / (lam * gm ** (lam - 1))
        return _ss_res(X, w)[0]

    grid = np.round(np.linspace(-3, 3, 241), 6)
    lnss = np.log([max(ss(l), 1e-300) for l in grid])
    i = int(np.argmin(lnss))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(lambda l: math.log(max(ss(l), 1e-300)), bounds=(lo, hi),
                                   method="bounded")
    best = float(res.x)
    ss_min = ss(best)
    df = len(y) - np.linalg.matrix_rank(X)
    t = stats.t.ppf(0.5 + level / 2, df)
    cut = math.log(ss_min * (1 + t ** 2 / df))

    def edge(a, b):
        fa = math.log(ss(a)) - cut
        fb = math.log(ss(b)) - cut
        if fa * fb > 0:
            return None
        return optimize.brentq(lambda l: math.log(ss(l)) - cut, a, b)

    low = edge(-3.0, best) if math.log(ss(-3.0)) > cut else None
    high = edge(best, 3.0) if math.log(ss(3.0)) > cut else None
    in_ci = (low is None or low <= current) and (high is None or current <= high)
    standard = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    ok = [s for s in standard if (low is None or s >= low) and (high is None or s <= high)]
    rec = min(ok or standard, key=lambda s: abs(s - best))
    if (low is None or low <= 1) and (high is None or 1 <= high):
        rec = 1
    names = {1: "None", 0.5: "Square Root", 0: "Log", -0.5: "Inverse Sqrt", -1: "Inverse"}
    return {"lambdas": grid, "ln_ss": lnss, "best": best, "low": low, "high": high,
            "cut": cut, "current": current, "current_in_ci": in_ci, "recommended": rec,
            "recommended_name": names.get(rec, f"Power (lambda = {rec:g})"),
            "ln_ss_current": math.log(ss(current)), "k": k}


# ---------------------------------------------------------------------------- effects
def factorial_effects(space: Space, obs: Observations, max_order: int | None = None,
                      transform: Transform | None = None) -> list[dict]:
    """Effects (= 2 x coded coefficient) of a two-level design with Lenth's PSE t-values."""
    transform = transform or Transform()
    z = transform.apply(obs.y)
    k = space.k
    max_order = max_order or min(k, 6)
    terms = []
    for r in range(1, max_order + 1):
        terms += list(itertools.combinations(range(k), r))
    kept = []
    for t in terms:
        mat = build_matrix(space, obs.coded, kept + [t], obs.blocks)
        if np.linalg.matrix_rank(mat.X) == mat.X.shape[1] and mat.X.shape[1] < len(z) + 1:
            kept.append(t)
        if mat.X.shape[1] >= len(z):
            break
    mat = build_matrix(space, obs.coded, kept, obs.blocks)
    beta, *_ = np.linalg.lstsq(mat.X, z, rcond=None)
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    effects = []
    for t, s in mat.groups:
        if not t:
            continue
        b = float(beta[s][0])
        col = mat.X[:, s][:, 0]
        ss = b ** 2 * float(col @ col)
        effects.append({"term": t, "name": term_name(t), "effect": 2 * b, "coef": b,
                        "ss": ss, "contribution": 100 * ss / ss_tot if ss_tot else 0.0})
    for e, (t, s) in zip(effects, [g for g in mat.groups if g[0]]):
        e["column"] = mat.X[:, s][:, 0]
    return lenth(effects)


def lenth(effects: list[dict]) -> list[dict]:
    """Lenth's (1989) pseudo standard error, t-values and the t / Bonferroni limits (m/3 df)
    of a set of effects (updated in place)."""
    abs_e = np.abs([e["effect"] for e in effects])
    m = len(abs_e)
    if m:
        s0 = 1.5 * np.median(abs_e)
        trimmed = abs_e[abs_e < 2.5 * s0]
        pse = 1.5 * np.median(trimmed) if len(trimmed) else s0
        dfl = m / 3
        for e in effects:
            e["t"] = abs(e["effect"]) / pse if pse > 0 else float("nan")
        t_lim = stats.t.ppf(0.975, dfl)
        bonf = stats.t.ppf(1 - 0.05 / (2 * m), dfl)
        for e in effects:
            e["pse"], e["t_limit"], e["bonferroni"] = pse, t_lim, bonf
    return effects


def split_plot_effects(space: Space, obs: Observations, max_order: int | None = None,
                       transform: Transform | None = None) -> dict:
    """Effects of a two-level split-plot factorial, separated as in Stat-Ease's split-plot
    tutorial: whole-plot effects (contrasts constant within every group — the hard-to-change
    factors, their interactions and the easy-to-change interactions that split the whole
    plots) and subplot effects (all others). Each set gets its own Lenth PSE and half-normal
    plot, because whole-plot effects are estimated against the larger between-group
    variation. Returns {"whole": [...], "sub": [...]}."""
    if obs.groups is None:
        raise ValueError("Whole-plot and subplot effects need the Group column.")
    eff = factorial_effects(space, obs, max_order, transform)
    gidx = np.unique(obs.groups, return_inverse=True)[1]
    lo = np.full(gidx.max() + 1, np.inf)
    hi = np.full(gidx.max() + 1, -np.inf)
    whole, sub = [], []
    for e in eff:
        col = e["column"]
        np.minimum.at(lo, gidx, col)
        np.maximum.at(hi, gidx, col)
        is_whole = bool(np.all(hi - lo < 1e-9))
        lo[:], hi[:] = np.inf, -np.inf
        e["whole_plot"] = is_whole
        (whole if is_whole else sub).append(dict(e))
    return {"whole": lenth(whole), "sub": lenth(sub)}


def is_two_level(space: Space, obs: Observations) -> bool:
    if space.mixture or any(space.categoric):
        return False
    pts = obs.coded[~np.all(np.abs(obs.coded) < 1e-9, axis=1)]
    return len(pts) > 0 and bool(np.all(np.abs(np.abs(pts) - 1) < 1e-6))


# ---------------------------------------------------------------------------- top level
@dataclass
class AnalysisSpec:
    response: str
    factors: list[str] | None = None
    model: str = "suggested"       # suggested | <order> | manual
    terms: list[str] | None = None  # term keys for manual
    reduction: str = "none"        # none | backward | forward | stepwise
    alpha_out: float = 0.10
    hierarchy: bool = True
    transform: Transform = field(default_factory=Transform)
    criterion: str = "p"           # p | aicc | bic (Auto Select criterion)
    alpha_in: float = 0.10
    selection: dict | None = None  # last Auto Select settings shown on the Model tab

    def to_meta(self) -> dict:
        d = {"model": self.model, "terms": self.terms, "reduction": self.reduction,
             "alpha_out": self.alpha_out, "hierarchy": self.hierarchy,
             "transform": self.transform.to_meta(), "factors": self.factors,
             "criterion": self.criterion, "alpha_in": self.alpha_in}
        if self.selection:
            d["selection"] = dict(self.selection)
        return d

    @classmethod
    def from_meta(cls, response: str, d: dict | None) -> "AnalysisSpec":
        d = d or {}
        return cls(response, d.get("factors"), d.get("model", "suggested"), d.get("terms"),
                   d.get("reduction", "none"), float(d.get("alpha_out", 0.10)),
                   bool(d.get("hierarchy", True)), Transform.from_meta(d.get("transform")),
                   d.get("criterion", "p"), float(d.get("alpha_in", 0.10)),
                   d.get("selection"))


@dataclass
class Analysis:
    fit: Fit
    summary: FitSummary
    spec: AnalysisSpec
    steps: list
    dropped: list
    space: Space
    obs: Observations


def response_name(ds, column: str) -> str:
    for r in (ds.meta.get("doe") or {}).get("responses", []):
        if r.get("column", "").lower() == column.lower():
            return r.get("name") or column
    return column


def response_units(ds, column: str) -> str:
    for r in (ds.meta.get("doe") or {}).get("responses", []):
        if r.get("column", "").lower() == column.lower():
            return r.get("units", "")
    return ""


def stored_spec(ds, response: str) -> AnalysisSpec:
    meta = (ds.meta.get("doe") or {}).get("analysis", {})
    return AnalysisSpec.from_meta(response, meta.get(ds.var(response).name))


def analyze(ds, spec: AnalysisSpec, space: Space | None = None) -> Analysis:
    space = space or space_from_dataset(ds, spec.factors)
    if space.k == 0:
        raise ValueError("Select at least one factor.")
    obs = observations(ds, space, spec.response)
    if len(obs.y) < 2:
        raise ValueError(f"Response {spec.response} has fewer than 2 values. Enter the "
                         "measured responses in the Data Editor first.")
    summary = fit_summary(space, obs, spec.transform)
    if spec.model == "manual" and spec.terms is not None:
        terms = [key_term(k) for k in spec.terms]
    elif spec.model == "suggested":
        terms = model_terms(space, summary.suggested_order)
    else:
        terms = model_terms(space, spec.model)
    terms, dropped = drop_aliased(space, obs, terms)
    steps: list = []
    if spec.reduction in ("backward", "forward", "stepwise"):
        fit, steps = auto_select(space, obs, terms, spec.transform, spec.reduction,
                                 spec.criterion, spec.alpha_in, spec.alpha_out,
                                 spec.hierarchy, spec.response)
    else:
        fit = fit_class(space)(space, obs, terms, spec.transform, spec.response)
    fit.response_name = response_name(ds, spec.response)
    fit.response_units = response_units(ds, spec.response)
    return Analysis(fit, summary, spec, steps, dropped, space, obs)


def save_spec(ds, spec: AnalysisSpec, fit: Fit | None = None) -> None:
    """Remember the chosen model for Model Graphs / Optimization / Point Prediction."""
    meta = ds.meta.setdefault("doe", {})
    d = spec.to_meta()
    if fit is not None:
        d["terms"] = [term_key(t) for t in fit.terms]
        d["model"] = "manual"
    meta.setdefault("analysis", {})[ds.var(spec.response).name] = d

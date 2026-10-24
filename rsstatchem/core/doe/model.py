"""Response surface model fitting as in Design-Expert's analysis node.

Covers factor coding (numeric factors to -1/+1, categoric factors effect-coded, mixture
components to L-pseudo components), polynomial and Scheffé mixture model terms, least squares,
the Fit Summary (sequential model sums of squares, lack-of-fit tests, model summary statistics,
suggested model), the ANOVA with partial (Type III) sums of squares, lack of fit and pure
error, fit statistics (Std. Dev., Mean, C.V. %, R², Adjusted R², Predicted R², PRESS, Adeq
Precision), coefficient estimates in coded terms with SE, CI and VIF, final equations in coded
and actual factors, response transformations including Box-Cox, backward reduction respecting
hierarchy, 2-level factorial effects (half-normal / Pareto with Lenth's PSE) and case
diagnostics (externally studentized residuals, leverage, Cook's distance, DFFITS).
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
                          index=j)
        else:
            low = m.get("low", float(x.min()) if len(x) else -1.0)
            high = m.get("high", float(x.max()) if len(x) else 1.0)
            if ftype == "mixture" and "low" not in m:
                low = 0.0
            f = FactorDef(m.get("name") or var.name, var.name, m.get("units", ""), ftype,
                          float(low), float(high), index=j)
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
    return Space(factors, kind, total, block, run, std)


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


def observations(ds, space: Space, response: str) -> Observations:
    from rsstatchem.core.stats.common import numeric

    frame = pd.DataFrame({f"f{j}": numeric(ds, f.column) for j, f in enumerate(space.factors)})
    frame["y"] = numeric(ds, response)
    if space.block_col:
        frame["block"] = numeric(ds, space.block_col)
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
    return Observations(actual, coded, frame["y"].to_numpy(float), blocks,
                        frame["run"].to_numpy(float), frame["std"].to_numpy(float),
                        frame.index.to_numpy())


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
def _effect_codes(idx: np.ndarray, nlev: int) -> np.ndarray:
    idx = idx.astype(int)
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
            suffixes = [s + f"[{j + 1}]" for s in suffixes for j in range(E.shape[1])]
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
    return _effect_codes(idx, len(levels)), levels


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
        self.h = np.einsum("ij,jk,ik->i", X, self.XtX_inv, X)
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
        s = math.sqrt(self.ms_res)
        h = np.clip(self.h, 0, 1 - 1e-12)
        r = self.resid / (s * np.sqrt(1 - h))
        df = self.df_res
        with np.errstate(invalid="ignore", divide="ignore"):
            t = r * np.sqrt((df - 1) / np.clip(df - r ** 2, 1e-12, None)) if df > 1 else r * 0
        cook = r ** 2 * h / (self.p * (1 - h))
        dffits = t * np.sqrt(h / (1 - h))
        return {"run": self.obs.run, "std": self.obs.std, "actual": self.z,
                "predicted": self.fitted, "residual": self.resid, "leverage": self.h,
                "int_stud": r, "ext_stud": t, "cooks": cook, "dffits": dffits}

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
        return R @ self.beta, np.einsum("ij,jk,ik->i", R, self.XtX_inv, R)

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
    kept, dropped = [], []
    for t in sort_terms(terms):
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


def backward(space: Space, obs: Observations, terms: list[Term], transform: Transform,
             alpha: float = 0.1, hierarchy: bool = True, response: str = "R1"):
    """Backward elimination by p-value (alpha out), keeping the model hierarchical."""
    terms = list(terms)
    steps = []
    while True:
        fit = Fit(space, obs, terms, transform, response)
        pv = fit.term_p_values()
        cands = [(pv[t][1], t) for t in terms if t in pv and removable(t, terms, space,
                                                                       hierarchy)]
        cands = [c for c in cands if np.isfinite(c[0])]
        if not cands:
            return fit, steps
        p, worst = max(cands, key=lambda c: (c[0], term_degree(c[1])))
        if p <= alpha:
            return fit, steps
        terms.remove(worst)
        steps.append((worst, p))


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
    reduction: str = "none"        # none | backward
    alpha_out: float = 0.10
    hierarchy: bool = True
    transform: Transform = field(default_factory=Transform)

    def to_meta(self) -> dict:
        return {"model": self.model, "terms": self.terms, "reduction": self.reduction,
                "alpha_out": self.alpha_out, "hierarchy": self.hierarchy,
                "transform": self.transform.to_meta(), "factors": self.factors}

    @classmethod
    def from_meta(cls, response: str, d: dict | None) -> "AnalysisSpec":
        d = d or {}
        return cls(response, d.get("factors"), d.get("model", "suggested"), d.get("terms"),
                   d.get("reduction", "none"), float(d.get("alpha_out", 0.10)),
                   bool(d.get("hierarchy", True)), Transform.from_meta(d.get("transform")))


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
    if spec.reduction == "backward":
        fit, steps = backward(space, obs, terms, spec.transform, spec.alpha_out,
                              spec.hierarchy, spec.response)
    else:
        fit = Fit(space, obs, terms, spec.transform, spec.response)
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

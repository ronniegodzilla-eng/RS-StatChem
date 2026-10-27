"""General Linear Model engine shared by GLM Univariate, Multivariate and Repeated Measures.

Mirrors SPSS GLM (UNIANOVA / GLM): a custom model built from terms (main effects,
interactions, factor-by-covariate and covariate-by-covariate products, nested effects such as
``B(A)``), with or without intercept, Type I–IV sums of squares, WLS weights, and the
quantities the SPSS sub-dialogs produce: contrasts (deviation, simple, difference, Helmert,
repeated, polynomial), estimated marginal means with pairwise comparisons and simple main
effects, parameter estimates (with HC0–HC4 robust standard errors), observed power, lack of
fit, heteroskedasticity tests, expected mean squares with Satterthwaite error terms for random
factors, and saved predicted values / residuals / diagnostics.

Design conventions follow SPSS:

* Type III tests use sum-to-zero (effect) coding; nested effects are coded within each cell of
  the nesting factors. Without empty cells this equals SPSS's Type III (and Type IV).
* Types I and II compare residual SSCP matrices of nested models built from overparameterized
  indicator columns (Type II: adjusted for every effect that does not contain the effect).
* Parameter estimates use the overparameterized design with later aliased columns set to 0
  ("This parameter is set to zero because it is redundant"), i.e. last category = reference.

Everything is a pure function of (data arrays, options) so that a later Bootstrap pass can
resample cases and call the same functions.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ------------------------------------------------------------------ term specification


@dataclass(frozen=True)
class TermSpec:
    """One model effect. ``factors`` are crossed factors, ``within`` the nesting factors (the
    effect ``B(A)`` has factors=("B",), within=("A",)), ``covariates`` the covariates
    multiplied in. Names are variable names as shown in SPSS output."""
    factors: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    within: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        parts = list(self.factors) + list(self.covariates)
        text = " * ".join(parts)
        if self.within:
            text += "(" + " * ".join(self.within) + ")"
        return text

    @property
    def syntax(self) -> str:
        return self.label.replace(" * ", "*")

    @property
    def all_factors(self) -> frozenset:
        return frozenset(self.factors) | frozenset(self.within)

    @property
    def is_pure_factor(self) -> bool:
        return not self.covariates

    def contains(self, other: "TermSpec") -> bool:
        """SPSS containment: same covariates, more factors, all factors of ``other`` in
        this effect."""
        return (sorted(self.covariates) == sorted(other.covariates)
                and other.all_factors < self.all_factors)


INTERCEPT = TermSpec()


def full_factorial(factors: list[str], covariates: list[str]) -> list[TermSpec]:
    """SPSS default design: covariates, then all factor main effects and interactions."""
    terms = [TermSpec(covariates=(c,)) for c in covariates]
    for r in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, r):
            terms.append(TermSpec(factors=combo))
    return terms


def main_effects(factors: list[str], covariates: list[str]) -> list[TermSpec]:
    return [TermSpec(covariates=(c,)) for c in covariates] + \
        [TermSpec(factors=(f,)) for f in factors]


def interactions_of(names: list[str], kind: str, factors: set[str]) -> list[TermSpec]:
    """Build terms from selected variables like SPSS's Build Term(s) drop-down:
    'interaction', 'main', 'all2' .. 'all5'."""
    out = []

    def make(combo):
        f = tuple(n for n in combo if n in factors)
        c = tuple(n for n in combo if n not in factors)
        return TermSpec(factors=f, covariates=c)

    if kind == "main":
        out = [make((n,)) for n in names]
    elif kind == "interaction":
        out = [make(tuple(names))] if names else []
    elif kind.startswith("all"):
        r = int(kind[3:])
        out = [make(c) for c in itertools.combinations(names, r)] if len(names) >= r else []
    return out


_NAME = r"[^\s\*\(\)]+"


def parse_terms(text: str, factors: list[str], covariates: list[str]) -> list[TermSpec]:
    """Parse an SPSS /DESIGN list such as ``A B A*B X A*X C(A)`` or ``C WITHIN A``."""
    lookup = {n.lower(): n for n in list(factors) + list(covariates)}
    fset = {f.lower() for f in factors}
    text = re.sub(r"\s+WITHIN\s+", "WITHIN", text.strip(), flags=re.IGNORECASE)
    terms = []
    for tok in text.replace(",", " ").split():
        within = ()
        if "WITHIN" in tok:
            left, right = tok.split("WITHIN", 1)
            tok, within = left, tuple(right.split("*"))
        m = re.fullmatch(r"(.+?)\((.+)\)", tok)
        if m:
            tok, within = m.group(1), tuple(m.group(2).split("*"))
        parts = [p for p in tok.split("*") if p]
        names = []
        for p in parts + list(within):
            if p.lower() not in lookup:
                raise ValueError(f"Unknown variable in model term: {p}")
            names.append(lookup[p.lower()])
        main = names[:len(parts)]
        wnames = tuple(names[len(parts):])
        if any(w.lower() not in fset for w in wnames):
            raise ValueError("Only factors can be nesting variables.")
        terms.append(TermSpec(tuple(n for n in main if n.lower() in fset),
                              tuple(n for n in main if n.lower() not in fset), wnames))
    return terms


def term_to_text(term: TermSpec) -> str:
    return term.syntax


# ------------------------------------------------------------------ design building
@dataclass
class FactorInfo:
    name: str
    levels: list           # raw codes, sorted
    labels: list[str]      # value labels for output
    codes: np.ndarray      # level index per case


def _contrast(codes: np.ndarray, m: int) -> np.ndarray:
    cols = np.zeros((len(codes), max(m - 1, 0)))
    for j in range(m - 1):
        cols[:, j] = (codes == j).astype(float) - (codes == m - 1).astype(float)
    return cols


def _indicator(codes: np.ndarray, m: int) -> np.ndarray:
    return np.column_stack([(codes == j).astype(float) for j in range(m)]) if m else \
        np.zeros((len(codes), 0))


def _rowprod(blocks: list[np.ndarray], n: int) -> np.ndarray:
    out = np.ones((n, 1))
    for b in blocks:
        out = np.einsum("ni,nj->nij", out, b).reshape(n, -1)
    return out


def _plain(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


class Design:
    """Builds effect-coded (Type III / EMM) and indicator (Types I, II, parameters) design
    matrices for a list of terms, for the observed cases or for synthetic cells."""

    def __init__(self, factors: list[FactorInfo], covariates: dict[str, np.ndarray],
                 terms: list[TermSpec], intercept: bool = True, n: int | None = None):
        self.factors = factors
        self.fidx = {f.name: i for i, f in enumerate(factors)}
        self.covariates = covariates
        self.terms = list(terms)
        self.intercept = intercept
        if n is None:
            n = len(next(iter(covariates.values()))) if covariates else \
                (len(factors[0].codes) if factors else 0)
        self.n = n
        self.cov_means = {k: float(np.mean(v)) for k, v in covariates.items()}
        # nested terms: present level combinations of the nested factors within each
        # combination of nesting factors (from the data)
        self.nest: dict[TermSpec, list[tuple[tuple, list[tuple]]]] = {}
        codes = self.code_matrix()
        for t in self.terms:
            if not t.within:
                continue
            wi = [self.fidx[w] for w in t.within]
            fi = [self.fidx[f] for f in t.factors]
            cells: dict[tuple, set] = {}
            for row in codes:
                cells.setdefault(tuple(row[wi]), set()).add(tuple(row[fi]))
            self.nest[t] = [(w, sorted(cells[w])) for w in sorted(cells)]
        # no-intercept: the first one-factor main effect absorbs the intercept
        self.absorb = None
        if not intercept:
            for t in self.terms:
                if len(t.factors) == 1 and not t.covariates and not t.within:
                    self.absorb = t
                    break

    def code_matrix(self) -> np.ndarray:
        if not self.factors:
            return np.zeros((self.n, 0), int)
        return np.column_stack([f.codes for f in self.factors])

    # ---------------------------------------------------------------- columns of a term
    def _cov_block(self, t: TermSpec, covs: dict[str, np.ndarray], n: int) -> np.ndarray:
        x = np.ones((n, 1))
        for c in t.covariates:
            x = x * np.asarray(covs[c], float)[:, None]
        return x

    def term_effect(self, t: TermSpec, codes: np.ndarray, covs: dict[str, np.ndarray]
                    ) -> np.ndarray:
        n = codes.shape[0]
        cov = self._cov_block(t, covs, n)
        if t.within:
            wi = [self.fidx[w] for w in t.within]
            fi = [self.fidx[f] for f in t.factors]
            blocks = []
            for w, present in self.nest[t]:
                inside = np.all(codes[:, wi] == np.array(w), axis=1).astype(float)
                if len(present) < 2:
                    continue
                combo = [tuple(r) for r in codes[:, fi]]
                last = present[-1]
                for p in present[:-1]:
                    col = np.array([(c == p) - (c == last) for c in combo], float)
                    blocks.append((inside * col)[:, None])
            out = np.hstack(blocks) if blocks else np.zeros((n, 0))
            return out * cov
        mats = []
        for f in t.factors:
            fi = self.fidx[f]
            m = len(self.factors[fi].levels)
            mats.append(_indicator(codes[:, fi], m) if t == self.absorb
                        else _contrast(codes[:, fi], m))
        return _rowprod(mats, n) * cov

    def term_indicator(self, t: TermSpec, codes: np.ndarray, covs: dict[str, np.ndarray]
                       ) -> tuple[np.ndarray, list[str]]:
        n = codes.shape[0]
        cov = self._cov_block(t, covs, n)
        ctext = "".join(f" * {c}" for c in t.covariates)
        if t.within:
            wi = [self.fidx[w] for w in t.within]
            fi = [self.fidx[f] for f in t.factors]
            cols, labels = [], []
            for w, present in self.nest[t]:
                inside = np.all(codes[:, wi] == np.array(w), axis=1)
                wl = " * ".join(f"[{nm}={_plain(self.factors[self.fidx[nm]].levels[lv])}]"
                                for nm, lv in zip(t.within, w))
                for p in present:
                    col = inside & np.all(codes[:, fi] == np.array(p), axis=1)
                    cols.append(col.astype(float))
                    fl = " * ".join(f"[{nm}={_plain(self.factors[self.fidx[nm]].levels[lv])}]"
                                    for nm, lv in zip(t.factors, p))
                    labels.append(f"{fl}{ctext}({wl})")
            X = np.column_stack(cols) if cols else np.zeros((n, 0))
            return X * cov, labels
        if not t.factors:
            return cov, [t.label]
        mats = [_indicator(codes[:, self.fidx[f]], len(self.factors[self.fidx[f]].levels))
                for f in t.factors]
        X = _rowprod(mats, n) * cov
        labels = []
        for combo in itertools.product(*[range(len(self.factors[self.fidx[f]].levels))
                                         for f in t.factors]):
            labels.append(" * ".join(
                f"[{f}={_plain(self.factors[self.fidx[f]].levels[lv])}]"
                for f, lv in zip(t.factors, combo)) + ctext)
        return X, labels

    # ---------------------------------------------------------------- whole designs
    def effect_matrix(self, codes: np.ndarray | None = None,
                      covs: dict[str, np.ndarray] | None = None
                      ) -> tuple[np.ndarray, list[list[int]]]:
        """Effect-coded design and the column indices of [Intercept] + terms."""
        codes = self.code_matrix() if codes is None else codes
        covs = self.covariates if covs is None else covs
        n = codes.shape[0]
        blocks, idx, pos = [], [], 0
        if self.intercept:
            blocks.append(np.ones((n, 1)))
            idx.append([0])
            pos = 1
        else:
            idx.append([])
        for t in self.terms:
            X = self.term_effect(t, codes, covs)
            blocks.append(X)
            idx.append(list(range(pos, pos + X.shape[1])))
            pos += X.shape[1]
        return (np.hstack(blocks) if blocks else np.zeros((n, 0))), idx

    def indicator_matrix(self) -> tuple[np.ndarray, list[str], list[int]]:
        """Overparameterized design: columns, parameter labels and term number per column
        (0 = intercept, i = terms[i-1])."""
        codes = self.code_matrix()
        blocks, labels, owner = [], [], []
        if self.intercept:
            blocks.append(np.ones((self.n, 1)))
            labels.append("Intercept")
            owner.append(0)
        for k, t in enumerate(self.terms, start=1):
            X, lab = self.term_indicator(t, codes, self.covariates)
            blocks.append(X)
            labels += lab
            owner += [k] * X.shape[1]
        X = np.hstack(blocks) if blocks else np.zeros((self.n, 0))
        return X, labels, owner

    def cell_row(self, levels: dict[int, int]) -> np.ndarray:
        """Effect-coded row for one cell (factor index -> level index) with covariates at
        their means; factors not given must not appear in the model."""
        codes = np.array([[levels.get(i, 0) for i in range(len(self.factors))]], int) \
            if self.factors else np.zeros((1, 0), int)
        covs = {k: np.array([v]) for k, v in self.cov_means.items()}
        return self.effect_matrix(codes, covs)[0][0]

    def average_row(self, fixed: dict[int, int]) -> np.ndarray:
        """Unweighted average of cell rows over all factors not fixed (SPSS EMMEANS)."""
        ranges = [[fixed[i]] if i in fixed else range(len(f.levels))
                  for i, f in enumerate(self.factors)]
        cells = list(itertools.product(*ranges))
        if not cells:
            return self.cell_row({})
        codes = np.array(cells, int).reshape(len(cells), len(self.factors))
        covs = {k: np.full(len(cells), v) for k, v in self.cov_means.items()}
        X, _ = self.effect_matrix(codes, covs)
        return X.mean(axis=0)


def aliased_columns(X: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Boolean mask of columns that are linear combinations of earlier columns (the SPSS
    sweep order), using modified Gram-Schmidt."""
    n, k = X.shape
    Q = np.zeros((n, 0))
    redundant = np.zeros(k, bool)
    for j in range(k):
        v = X[:, j].astype(float)
        norm0 = np.linalg.norm(v)
        if norm0 == 0:
            redundant[j] = True
            continue
        for _ in range(2):
            v = v - Q @ (Q.T @ v)
        if np.linalg.norm(v) <= tol * max(norm0, 1.0) * 1e3:
            redundant[j] = True
        else:
            Q = np.column_stack([Q, v / np.linalg.norm(v)])
    return redundant


# ------------------------------------------------------------------ the fitted model
@dataclass
class Term:
    """A fitted model effect (compatible with :class:`manova.LinearModel` users)."""
    label: str
    idx: list[int]
    factors: tuple[int, ...] = ()
    covariate: str | None = None
    spec: TermSpec = field(default_factory=TermSpec)


class GLMFit:
    """Multivariate (W)LS fit ``Y = X B + E`` with SPSS's coding and SS types."""

    def __init__(self, Y: np.ndarray, factors: list[FactorInfo],
                 covariates: dict[str, np.ndarray], terms: list[TermSpec] | None = None,
                 intercept: bool = True, ss_type: int = 3, weights: np.ndarray | None = None):
        self.Y = np.asarray(Y, float)
        if self.Y.ndim == 1:
            self.Y = self.Y[:, None]
        self.n = self.Y.shape[0]
        self.factors = factors
        self.covariates = covariates
        if terms is None:
            terms = full_factorial([f.name for f in factors], list(covariates))
        self.specs = list(terms)
        self.intercept = intercept
        self.ss_type = int(ss_type)
        self.w = np.ones(self.n) if weights is None else np.asarray(weights, float)
        self.sw = np.sqrt(self.w)
        self.design = Design(factors, covariates, self.specs, intercept, self.n)
        self.cov_means = self.design.cov_means
        X, idx = self.design.effect_matrix()
        self.X = X
        self.terms = []
        if intercept:
            self.terms.append(Term("Intercept", idx[0], (), None, INTERCEPT))
        fpos = {f.name: i for i, f in enumerate(factors)}
        for t, cols in zip(self.specs, idx[1:]):
            self.terms.append(Term(t.label, cols,
                                   tuple(fpos[f] for f in list(t.factors) + list(t.within)),
                                   t.covariates[0] if t.covariates and not t.factors
                                   and len(t.covariates) == 1 else None, t))
        Xw = X * self.sw[:, None]
        Yw = self.Y * self.sw[:, None]
        self.Xw, self.Yw = Xw, Yw
        self.XtX_inv = np.linalg.pinv(Xw.T @ Xw)
        self.B = self.XtX_inv @ Xw.T @ Yw
        self.fitted = X @ self.B
        self.resid = self.Y - self.fitted
        rw = self.resid * self.sw[:, None]
        self.E = rw.T @ rw
        self.rank = int(np.linalg.matrix_rank(Xw)) if X.shape[1] else 0
        self.ve = self.n - self.rank
        self.deficient = self.rank < X.shape[1]
        # overparameterized design for Types I/II and parameter estimates
        Xo, self.param_labels, self.param_owner = self.design.indicator_matrix()
        self.Xo = Xo
        self.redundant = aliased_columns(Xo * self.sw[:, None]) if Xo.shape[1] else \
            np.zeros(0, bool)
        self._sscp_cache: dict[tuple, np.ndarray] = {}

    @property
    def design_text(self) -> str:
        return " + ".join(t.label for t in self.terms)

    # ---------------------------------------------------------------- SSCP of submodels
    def _resid_sscp(self, owners: tuple[int, ...]) -> tuple[np.ndarray, int]:
        key = tuple(sorted(owners))
        if key in self._sscp_cache:
            return self._sscp_cache[key]
        cols = [j for j, o in enumerate(self.param_owner) if o in set(key)]
        if cols:
            Xs = self.Xo[:, cols] * self.sw[:, None]
            beta, *_ = np.linalg.lstsq(Xs, self.Yw, rcond=None)
            r = self.Yw - Xs @ beta
            rank = int(np.linalg.matrix_rank(Xs))
        else:
            r, rank = self.Yw, 0
        out = (r.T @ r, rank)
        self._sscp_cache[key] = out
        return out

    def term_number(self, term: Term) -> int:
        """0 for the intercept, i for specs[i-1]."""
        if term.label == "Intercept" and term.spec == INTERCEPT:
            return 0
        return self.specs.index(term.spec) + 1

    def _reduced_sets(self, k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(model without term k, model with term k) as term-number tuples for Types I/II."""
        all_k = ([0] if self.intercept else []) + list(range(1, len(self.specs) + 1))
        spec = INTERCEPT if k == 0 else self.specs[k - 1]
        if self.ss_type == 1:
            before = tuple(all_k[:all_k.index(k)])
            return before, before + (k,)
        others = []
        for j in all_k:
            if j == k:
                continue
            sj = INTERCEPT if j == 0 else self.specs[j - 1]
            if k == 0:
                contains = sj.is_pure_factor and sj != INTERCEPT
            else:
                contains = sj.contains(spec)
            if not contains:
                others.append(j)
        return tuple(others), tuple(others) + (k,)

    def hypothesis(self, term: Term, M: np.ndarray | None = None) -> np.ndarray:
        """Hypothesis SSCP for ``term`` (in the model's SS type) on ``Y M``."""
        if self.ss_type in (1, 2):
            k = self.term_number(term)
            a, b = self._reduced_sets(k)
            Ea, _ = self._resid_sscp(a)
            Eb, _ = self._resid_sscp(b)
            H = Ea - Eb
            return H if M is None else M.T @ H @ M
        B = self.B if M is None else self.B @ M
        bt = B[term.idx, :]
        V = self.XtX_inv[np.ix_(term.idx, term.idx)]
        return bt.T @ np.linalg.pinv(V) @ bt

    def term_df(self, term: Term) -> int:
        if self.ss_type in (1, 2):
            k = self.term_number(term)
            a, b = self._reduced_sets(k)
            return self._resid_sscp(b)[1] - self._resid_sscp(a)[1]
        V = self.XtX_inv[np.ix_(term.idx, term.idx)]
        return int(np.linalg.matrix_rank(V)) if len(term.idx) else 0

    def error(self, M: np.ndarray | None = None) -> np.ndarray:
        return self.E if M is None else M.T @ self.E @ M

    # ---------------------------------------------------------------- marginal means
    def cell_row(self, levels) -> np.ndarray:
        if isinstance(levels, dict):
            return self.design.cell_row(levels)
        return self.design.cell_row(dict(enumerate(levels)))

    def average_row(self, fixed: dict[int, int]) -> np.ndarray:
        return self.design.average_row(fixed)

    def estimate(self, l: np.ndarray, c: np.ndarray) -> tuple[float, float]:
        est = float(l @ self.B @ c)
        sigma = self.E / self.ve if self.ve > 0 else self.E * np.nan
        var = float(l @ self.XtX_inv @ l) * float(c @ sigma @ c)
        return est, math.sqrt(max(var, 0.0))

    # ---------------------------------------------------------------- whole-model sums
    def model_ss(self) -> tuple[np.ndarray, int, np.ndarray, int]:
        """(corrected (or uncorrected) model SS per DV, its df, total SS per DV, total df)."""
        yw = self.Yw
        sse = np.diag(self.E)
        if self.intercept:
            mean = (self.w @ self.Y) / self.w.sum()
            sst = ((self.Y - mean) ** 2 * self.w[:, None]).sum(axis=0)
            return sst - sse, self.rank - 1, sst, self.n - 1
        sst = (yw ** 2).sum(axis=0)
        return sst - sse, self.rank, sst, self.n


# ------------------------------------------------------------------ power, contrasts
def observed_power_f(F: float, df1: float, df2: float, alpha: float = 0.05) -> tuple[float, float]:
    """(noncentrality parameter, observed power) as SPSS computes them for an F test."""
    if not (np.isfinite(F) and df1 > 0 and df2 > 0):
        return float("nan"), float("nan")
    lam = F * df1
    crit = stats.f.ppf(1 - alpha, df1, df2)
    return lam, float(stats.ncf.sf(crit, df1, df2, lam)) if lam > 0 else alpha


def observed_power_t(t: float, df: float, alpha: float = 0.05) -> tuple[float, float]:
    if not (np.isfinite(t) and df > 0):
        return float("nan"), float("nan")
    nc = abs(t)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return nc, float(stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc))


CONTRAST_TYPES = ("none", "deviation", "simple", "difference", "helmert", "repeated",
                  "polynomial")
CONTRAST_NAMES = {"deviation": "Deviation", "simple": "Simple", "difference": "Difference",
                  "helmert": "Helmert", "repeated": "Repeated", "polynomial": "Polynomial"}


def poly_coefficients(k: int, metric: list[float] | None = None) -> np.ndarray:
    """Orthonormal polynomial contrasts (k x (k-1)); ``metric`` gives the level spacing."""
    x = np.arange(1, k + 1, dtype=float) if metric is None else np.asarray(metric, float)
    V = np.vander(x - x.mean(), k, increasing=True)
    Q, R = np.linalg.qr(V)
    Q = Q * np.sign(np.diag(R))
    return Q[:, 1:]


def poly_names(k: int) -> list[str]:
    base = ["Linear", "Quadratic", "Cubic"]
    return [base[i] if i < 3 else f"Order {i + 1}" for i in range(k - 1)]


def contrast_matrix(kind: str, k: int, reference: str = "last") -> tuple[np.ndarray, list[str]]:
    """Rows = contrasts on the k level means, with SPSS's row labels."""
    ref = k - 1 if reference == "last" else 0
    rows, labels = [], []
    if kind == "deviation":
        for i in range(k):
            if i == ref:
                continue
            r = -np.ones(k) / k
            r[i] += 1
            rows.append(r)
            labels.append(f"Level {i + 1} vs. Mean")
    elif kind == "simple":
        for i in range(k):
            if i == ref:
                continue
            r = np.zeros(k)
            r[i], r[ref] = 1, -1
            rows.append(r)
            labels.append(f"Level {i + 1} vs. Level {ref + 1}")
    elif kind == "difference":
        for i in range(1, k):
            r = np.zeros(k)
            r[i] = 1
            r[:i] = -1 / i
            rows.append(r)
            labels.append(f"Level {i + 1} vs. Level 1" if i == 1 else
                          f"Level {i + 1} vs. Previous")
    elif kind == "helmert":
        for i in range(k - 1):
            r = np.zeros(k)
            r[i] = 1
            r[i + 1:] = -1 / (k - i - 1)
            rows.append(r)
            labels.append(f"Level {i + 1} vs. Level {k}" if i == k - 2 else
                          f"Level {i + 1} vs. Later")
    elif kind == "repeated":
        for i in range(k - 1):
            r = np.zeros(k)
            r[i], r[i + 1] = 1, -1
            rows.append(r)
            labels.append(f"Level {i + 1} vs. Level {i + 2}")
    elif kind == "polynomial":
        P = poly_coefficients(k)
        rows = list(P.T)
        labels = poly_names(k)
    else:
        raise ValueError(kind)
    return np.array(rows).reshape(len(rows), k), labels


# ------------------------------------------------------------------ diagnostics
def leverage(fit: GLMFit) -> np.ndarray:
    """Uncentered leverage h_ii of the (weighted) design."""
    Xw = fit.Xw
    return np.einsum("ij,jk,ik->i", Xw, fit.XtX_inv, Xw)


def saved_values(fit: GLMFit, j: int = 0) -> dict[str, np.ndarray]:
    """SPSS GLM /SAVE quantities for dependent variable j."""
    h = leverage(fit)
    mse = fit.E[j, j] / fit.ve if fit.ve > 0 else float("nan")
    e = fit.resid[:, j]
    ew = e * fit.sw
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "PRE": fit.fitted[:, j],
            "SEP": np.sqrt(h * mse) / fit.sw,
            "RES": e,
            "WRE": ew,
            "ZRE": ew / math.sqrt(mse),
            "SRE": ew / np.sqrt(mse * (1 - h)),
            "DRE": e / (1 - h),
            "COO": ew ** 2 * h / (fit.rank * mse * (1 - h) ** 2),
            "LEV": h,
        }
    return out


def lack_of_fit(fit: GLMFit, j: int = 0) -> tuple[float, int, float, int] | None:
    """(SS lack of fit, df, SS pure error, df) from replicated predictor combinations."""
    keys = [tuple(f.codes) for f in fit.factors]
    keys += [tuple(np.round(v, 10)) for v in fit.covariates.values()]
    if not keys:
        return None
    combos = list(zip(*keys))
    groups: dict[tuple, list[int]] = {}
    for i, c in enumerate(combos):
        groups.setdefault(c, []).append(i)
    y = fit.Y[:, j]
    ss_pe = 0.0
    for idx in groups.values():
        w = fit.w[idx]
        m = float(np.average(y[idx], weights=w))
        ss_pe += float(np.sum(w * (y[idx] - m) ** 2))
    df_pe = fit.n - len(groups)
    sse = float(fit.E[j, j])
    df_lof = fit.ve - df_pe
    if df_pe <= 0 or df_lof <= 0:
        return None
    return sse - ss_pe, df_lof, ss_pe, df_pe


def heteroskedasticity(fit: GLMFit, j: int = 0) -> dict[str, tuple]:
    """SPSS GLM tests with the predicted values as the variance model (White uses the
    design columns, their squares and cross products)."""
    e = fit.resid[:, j] * fit.sw
    n = fit.n
    e2 = e ** 2
    yhat = fit.fitted[:, j] * fit.sw
    Z = np.column_stack([np.ones(n), yhat])

    def aux(Z):
        b, *_ = np.linalg.lstsq(Z, e2, rcond=None)
        f = Z @ b
        ssr = float(np.sum((f - e2.mean()) ** 2))
        sse = float(np.sum((e2 - f) ** 2))
        r2 = ssr / float(np.sum((e2 - e2.mean()) ** 2)) if np.var(e2) > 0 else 0.0
        return ssr, sse, r2, int(np.linalg.matrix_rank(Z)) - 1

    ssr, sse_aux, r2, q = aux(Z)
    sig2 = float(np.sum(e2)) / n
    bp = ssr / (2 * sig2 ** 2)
    mbp = n * r2
    dfe = n - q - 1
    Fh = (ssr / q) / (sse_aux / dfe) if q > 0 and dfe > 0 and sse_aux > 0 else float("nan")
    # White: non-redundant design columns (without intercept), squares and cross products
    Xr = fit.Xo[:, ~fit.redundant] * fit.sw[:, None]
    Xr = Xr[:, [c for c in range(Xr.shape[1]) if np.ptp(Xr[:, c]) > 0]]
    cols = [np.ones(n)]
    for a in range(Xr.shape[1]):
        cols.append(Xr[:, a])
        for b in range(a, Xr.shape[1]):
            cols.append(Xr[:, a] * Xr[:, b])
    W = np.column_stack(cols)
    keep = ~aliased_columns(W)
    _, _, r2w, qw = aux(W[:, keep])
    return {"mbp": (mbp, q, float(stats.chi2.sf(mbp, q))),
            "bp": (bp, q, float(stats.chi2.sf(bp, q))),
            "f": (Fh, q, dfe, float(stats.f.sf(Fh, q, dfe))),
            "white": (n * r2w, qw, float(stats.chi2.sf(n * r2w, qw)))}


def parameter_estimates(fit: GLMFit, j: int = 0, hc: str | None = None
                        ) -> list[tuple[str, float, float, float, float, bool]]:
    """(label, B, SE, t, p, redundant) with the last-category-redundant parameterization;
    ``hc`` = 'HC0'..'HC4' for robust standard errors."""
    keep = ~fit.redundant
    Xk = fit.Xo[:, keep] * fit.sw[:, None]
    y = fit.Yw[:, j]
    XtXi = np.linalg.inv(Xk.T @ Xk)
    b = XtXi @ Xk.T @ y
    r = y - Xk @ b
    df = fit.n - Xk.shape[1]
    mse = float(r @ r) / df if df > 0 else float("nan")
    if hc:
        h = np.einsum("ij,jk,ik->i", Xk, XtXi, Xk)
        k = Xk.shape[1]
        if hc == "HC0":
            om = r ** 2
        elif hc == "HC1":
            om = r ** 2 * fit.n / df
        elif hc == "HC2":
            om = r ** 2 / (1 - h)
        elif hc == "HC3":
            om = r ** 2 / (1 - h) ** 2
        else:
            delta = np.minimum(4.0, fit.n * h / k)
            om = r ** 2 / (1 - h) ** delta
        cov = XtXi @ (Xk.T * om) @ Xk @ XtXi
    else:
        cov = XtXi * mse
    out = []
    pos = 0
    for label, red in zip(fit.param_labels, fit.redundant):
        if red:
            out.append((label, 0.0, float("nan"), float("nan"), float("nan"), True))
            continue
        se = math.sqrt(max(cov[pos, pos], 0.0))
        t = b[pos] / se if se > 0 else float("nan")
        p = float(2 * stats.t.sf(abs(t), df)) if se > 0 else float("nan")
        out.append((label, float(b[pos]), se, float(t), p, False))
        pos += 1
    return out


# ------------------------------------------------------------------ random effects (EMS)
def ems_table(fit: GLMFit, random_factors: list[str]) -> dict:
    """Expected mean squares (coefficients of the variance components) for every source,
    and the synthesized error terms with Satterthwaite df (SPSS UNIANOVA /RANDOM).

    Returns dict with 'random' (term numbers that are random), 'coef' {source k: {comp: c}}
    (comp = term number or 'error'), 'quad' {k: [labels]} and 'error' {k: (weights {comp: c},
    MS, df)}."""
    rset = set(random_factors)
    specs = [INTERCEPT] + fit.specs if fit.intercept else [None] + fit.specs
    numbers = ([0] if fit.intercept else []) + list(range(1, len(fit.specs) + 1))
    random_terms = [k for k in numbers if k > 0 and (set(specs[k].all_factors) & rset)]
    fixed_terms = [k for k in numbers if k not in random_terms]
    n = fit.n
    X, labels, owner = fit.Xo, fit.param_labels, fit.param_owner
    owner = np.array(owner)
    Z = {k: X[:, owner == k] for k in numbers}
    sw = fit.sw
    # quadratic-form matrices A_k with SS_k = y' A_k y (weighted scale)
    A = {}
    dfs = {}
    terms_by_k = {fit.term_number(t): t for t in fit.terms}
    for k in numbers:
        t = terms_by_k[k]
        if fit.ss_type in (1, 2):
            a, b = fit._reduced_sets(k)

            def proj(ks):
                cols = [j for j, o in enumerate(fit.param_owner) if o in set(ks)]
                if not cols:
                    return np.zeros((n, n))
                Xs = X[:, cols] * sw[:, None]
                return Xs @ np.linalg.pinv(Xs)
            A[k] = proj(b) - proj(a)
        else:
            L = np.zeros((len(t.idx), fit.X.shape[1]))
            for r, c in enumerate(t.idx):
                L[r, c] = 1
            G = fit.XtX_inv
            LGL = L @ G @ L.T
            K = fit.Xw @ G @ L.T
            A[k] = K @ np.linalg.pinv(LGL) @ K.T
        dfs[k] = max(int(round(np.trace(A[k]))), 1)
    coef, quad = {}, {}
    for k in numbers:
        c = {}
        for r in random_terms:
            Zr = Z[r] * sw[:, None]
            c[r] = float(np.trace(Zr.T @ A[k] @ Zr)) / dfs[k]
        c["error"] = float(np.trace(A[k])) / dfs[k]
        coef[k] = c
        q = []
        for f in fixed_terms:
            Xf = Z[f] * sw[:, None]
            if Xf.shape[1] and np.abs(Xf.T @ A[k] @ Xf).max() > 1e-8 * max(1.0, n):
                q.append("Intercept" if f == 0 else specs[f].label)
        quad[k] = q
    ms = {k: float(fit.hypothesis(terms_by_k[k])[0, 0]) / dfs[k] for k in numbers}
    ms["error"] = float(fit.E[0, 0]) / fit.ve if fit.ve > 0 else float("nan")
    dfm = dict(dfs)
    dfm["error"] = fit.ve
    errors = {}
    comps = random_terms + ["error"]
    for k in numbers:
        target = dict(coef[k])
        if k in random_terms:
            target[k] = 0.0
        support = {c for c in comps if abs(target.get(c, 0.0)) > 1e-9}
        cands = [r for r in random_terms if r != k
                 and {c for c in comps if abs(coef[r].get(c, 0.0)) > 1e-9} <= support]
        cands.append("error")
        Mx = np.array([[coef[r].get(c, 0.0) if r != "error" else (1.0 if c == "error" else 0.0)
                        for r in cands] for c in comps])
        tv = np.array([target.get(c, 0.0) for c in comps])
        w, *_ = np.linalg.lstsq(Mx, tv, rcond=None)
        weights = {r: float(x) for r, x in zip(cands, w) if abs(x) > 1e-8}
        msd = sum(x * ms[r] for r, x in weights.items())
        den = sum((x * ms[r]) ** 2 / dfm[r] for r, x in weights.items())
        df_d = msd ** 2 / den if den > 0 else float("nan")
        errors[k] = (weights, msd, df_d)
    return {"random": random_terms, "coef": coef, "quad": quad, "error": errors,
            "ms": ms, "df": dfm, "numbers": numbers, "labels": {
                k: ("Intercept" if k == 0 else specs[k].label) for k in numbers}}


__all__ = ["TermSpec", "INTERCEPT", "full_factorial", "main_effects", "parse_terms",
           "interactions_of", "FactorInfo", "Design", "GLMFit", "Term", "contrast_matrix",
           "poly_coefficients", "poly_names", "observed_power_f", "observed_power_t",
           "saved_values", "lack_of_fit", "heteroskedasticity", "parameter_estimates",
           "ems_table", "CONTRAST_TYPES", "CONTRAST_NAMES"]

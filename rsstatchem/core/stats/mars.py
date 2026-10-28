"""Analyze > Generalized Linear Models > Multiple Adaptive Regression Splines (SPSS 31,
STATS EARTH, which runs R's ``earth``). Own numpy implementation of Friedman's (1991) MARS
with earth's defaults, so results follow R ``earth``:

* **Forward pass**: starting from the intercept, add the pair of hinge functions
  B_p(x)·max(0, x − t) and B_p(x)·max(0, t − x) (parent term B_p, predictor x not yet in the
  parent, knot t at a data value) that reduces the residual sum of squares most; candidate
  knots skip ``endspan`` = ⌊3 − log2(0.05/p)⌋ observations at each end and are spaced
  ``minspan`` = ⌊−log2(−ln(0.95)/(p·n_parent))/2.5⌋ apart (Friedman's rules, as in earth).
  Stops when the R² gain is below ``thresh`` (0.001), R² ≥ 1 − thresh, or ``nk`` terms
  (min(200, max(20, 2p)) + 1) are reached. Fast search: candidate columns are orthogonalised
  against the current basis (QR), so every knot is scored without refitting.
* **Backward pass**: the term whose removal raises RSS least is dropped repeatedly; the subset
  with the smallest GCV = (RSS/N) / (1 − C/N)², C = M + penalty·(M − 1)/2 (penalty 2 for
  additive models, 3 with interactions) is selected.
* **Summary** (earth style): coefficients of the selected basis functions, GCV, RSS, GRSq,
  RSq, and variable importance (``evimp``: nsubsets, gcv and rss criteria scaled to 100).
* **Binary response**: like ``earth(glm = list(family = binomial))``, the basis is selected on
  the 0/1 response by least squares and the coefficients are then refitted by logistic
  regression (IRLS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import (MLError, MLResult, categorical, complete_rows,
                                             full_series, measure_of, raw_column)
from rsstatchem.core.stats.roc import figure, png, style


@dataclass
class Term:
    hinges: tuple = ()          # ((var, knot, sign), ...); sign +1 → max(0, x − t)
    linear: tuple = ()

    @property
    def degree(self) -> int:
        return len(self.hinges)

    def vars(self) -> set:
        return {h[0] for h in self.hinges}

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        out = np.ones(len(X))
        for v, t, s in self.hinges:
            out *= np.maximum(0.0, s * (X[:, v] - t))
        return out

    def label(self, names: list[str]) -> str:
        if not self.hinges:
            return "(Intercept)"
        parts = []
        for v, t, s in self.hinges:
            tt = f"{t:.6g}"
            parts.append(f"h({names[v]}-{tt})" if s > 0 else f"h({tt}-{names[v]})")
        return "*".join(parts)


@dataclass
class MarsModel:
    terms: list[Term]
    coef: np.ndarray
    names: list[str]
    rss: float
    gcv: float
    rsq: float
    grsq: float
    n: int
    penalty: float
    forward_terms: int
    termination: str
    selected: list[int]
    subsets: list = field(default_factory=list)    # (size, term indices, rss) per size
    all_terms: list = field(default_factory=list)
    glm: dict | None = None

    def basis(self, X):
        return np.column_stack([t.evaluate(X) for t in self.terms])

    def predict(self, X, type_: str = "link"):
        eta = self.basis(X) @ self.coef
        if self.glm is not None and type_ == "response":
            return 1 / (1 + np.exp(-eta))
        return eta


def _spans(n_used: int, p: int, minspan: int | None, endspan: int | None) -> tuple[int, int]:
    if endspan is None or endspan <= 0:
        endspan = int(3 - math.log2(0.05 / p))
    if minspan is None or minspan <= 0:
        minspan = int(-math.log2(-(1.0 / (p * max(n_used, 1))) * math.log(1 - 0.05)) / 2.5)
    return max(1, int(minspan)), max(0, int(endspan))


def _candidates(x: np.ndarray, parent: np.ndarray, minspan: int, endspan: int,
                existing=()) -> np.ndarray:
    """Knot values: eligible (parent > 0) cases sorted by x, endspan dropped at each end, one
    candidate every minspan observations counted from the top, ties at their last value, and
    no knot fewer than minspan observations away from a knot this parent already has on x."""
    xs = np.sort(x[parent > 0])
    m = len(xs)
    if m < 2 * endspan + 1:
        return np.array([])
    pos = np.arange(m - 2 - endspan, endspan - 1, -1)
    pos = pos[::minspan] if minspan > 1 else pos
    if len(existing):
        epos = np.searchsorted(xs, np.asarray(existing, float), side="right") - 1
        far = np.all(np.abs(pos[:, None] - epos[None, :]) > minspan, axis=1) if minspan > 1 \
            else ~np.isin(xs[pos], existing)
        pos = pos[far]
    vals = xs[pos]
    vals = vals[vals < xs[-1]]
    return np.unique(vals)


def _gcv(rss: float, n: int, nterms: int, penalty: float) -> float:
    c = nterms + penalty * (nterms - 1) / 2
    d = 1 - c / n
    return float(rss / n / (d * d)) if d > 0 else float("inf")


def forward_pass(X: np.ndarray, y: np.ndarray, degree: int = 1, nk: int | None = None,
                 thresh: float = 0.001, minspan: int | None = None,
                 endspan: int | None = None, penalty: float | None = None):
    n, p = X.shape
    nk = nk or min(200, max(20, 2 * p)) + 1
    terms = [Term()]
    B = np.ones((n, 1))
    tss = float(((y - y.mean()) ** 2).sum())
    rss = tss
    termination = ""
    used_knots: dict = {}
    while True:
        if len(terms) >= nk:
            termination = f"Reached nk {nk}"
            break
        Q, _ = np.linalg.qr(B)
        r = y - Q @ (Q.T @ y)
        best = None
        for pi, parent in enumerate(terms):
            if parent.degree >= degree:
                continue
            bp = B[:, pi]
            for v in range(p):
                if v in parent.vars():
                    continue
                ms, es = _spans(int((bp > 0).sum()), p, minspan, endspan)
                knots = _candidates(X[:, v], bp, ms, es, used_knots.get((pi, v), ()))
                if not len(knots):
                    continue
                xv = X[:, v][:, None]
                A1 = bp[:, None] * np.maximum(0.0, xv - knots[None, :])
                A2 = bp[:, None] * np.maximum(0.0, knots[None, :] - xv)
                P1 = A1 - Q @ (Q.T @ A1)
                P2 = A2 - Q @ (Q.T @ A2)
                g11 = (P1 * P1).sum(0)
                g22 = (P2 * P2).sum(0)
                g12 = (P1 * P2).sum(0)
                c1 = r @ P1
                c2 = r @ P2
                tol = 1e-10 * np.maximum((A1 * A1).sum(0) + (A2 * A2).sum(0), 1e-300)
                det = g11 * g22 - g12 ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    both = (c1 * c1 * g22 - 2 * c1 * c2 * g12 + c2 * c2 * g11) / det
                    one1 = np.where(g11 > tol, c1 * c1 / g11, 0.0)
                    one2 = np.where(g22 > tol, c2 * c2 / g22, 0.0)
                ok_both = (g11 > tol) & (g22 > tol) & (det > 1e-10 * g11 * g22)
                red = np.where(ok_both, both, np.maximum(one1, one2))
                red = np.nan_to_num(red, nan=0.0, neginf=0.0)
                k = int(np.argmax(red))
                if best is None or red[k] > best[0] * (1 + 1e-12):
                    use = "both" if ok_both[k] else ("pos" if one1[k] >= one2[k] else "neg")
                    best = (float(red[k]), pi, v, float(knots[k]), use)
        if best is None or best[0] <= 0:
            termination = "No new term increases RSq"
            break
        red, pi, v, t, use = best
        new_rss = rss - red
        rsq_old = 1 - rss / tss
        rsq_new = 1 - new_rss / tss
        parent = terms[pi]
        added = []
        if use in ("both", "pos"):
            added.append(Term(parent.hinges + ((v, t, +1),)))
        if use in ("both", "neg"):
            added.append(Term(parent.hinges + ((v, t, -1),)))
        if rsq_new - rsq_old < thresh:
            termination = f"RSq changed by less than {thresh:g} at {len(terms)} terms"
            break
        used_knots.setdefault((pi, v), []).append(t)
        for tm in added:
            terms.append(tm)
            B = np.column_stack([B, tm.evaluate(X)])
        rss = new_rss
        if rsq_new >= 1 - thresh:
            termination = f"Reached maximum RSq {1 - thresh:.4f} at {len(terms)} terms"
            break
    return terms, B, termination


def backward_pass(B: np.ndarray, y: np.ndarray, penalty: float):
    n, M = B.shape

    def rss_of(cols):
        coef = np.linalg.lstsq(B[:, cols], y, rcond=None)[0]
        e = y - B[:, cols] @ coef
        return float(e @ e)

    cur = list(range(M))
    subsets = [(len(cur), list(cur), rss_of(cur))]
    while len(cur) > 1:
        best = None
        for c in cur[1:]:
            trial = [k for k in cur if k != c]
            r = rss_of(trial)
            if best is None or r < best[0]:
                best = (r, trial)
        cur = best[1]
        subsets.append((len(cur), list(cur), best[0]))
    gcvs = [_gcv(r, n, size, penalty) for size, _, r in subsets]
    k = int(np.argmin(gcvs))
    return subsets, k


def _irls(B, y, max_iter=50):
    beta = np.zeros(B.shape[1])
    for _ in range(max_iter):
        eta = B @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        w = np.maximum(mu * (1 - mu), 1e-10)
        z = eta + (y - mu) / w
        new = np.linalg.lstsq(B * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None)[0]
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    eta = B @ beta
    mu = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
    eps = 1e-300
    dev = -2 * float(np.sum(y * np.log(np.maximum(mu, eps)) +
                            (1 - y) * np.log(np.maximum(1 - mu, eps))))
    p0 = y.mean()
    null = -2 * float(np.sum(y * math.log(max(p0, eps)) + (1 - y) * math.log(max(1 - p0, eps))))
    return beta, dev, null


def earth(X: np.ndarray, y: np.ndarray, names: list[str], degree: int = 1,
          nk: int | None = None, thresh: float = 0.001, minspan: int | None = None,
          endspan: int | None = None, penalty: float | None = None, nprune: int | None = None,
          binomial: bool = False) -> MarsModel:
    n, p = X.shape
    pen = penalty if penalty is not None else (2.0 if degree == 1 else 3.0)
    terms, B, termination = forward_pass(X, y, degree, nk, thresh, minspan, endspan, pen)
    subsets, k = backward_pass(B, y, pen)
    if nprune is not None:
        allowed = [i for i, s in enumerate(subsets) if s[0] <= nprune]
        gcvs = [_gcv(subsets[i][2], n, subsets[i][0], pen) for i in allowed]
        k = allowed[int(np.argmin(gcvs))]
    size, cols, rss = subsets[k]
    cols = sorted(cols)
    sel_terms = [terms[c] for c in cols]
    Bs = B[:, cols]
    coef = np.linalg.lstsq(Bs, y, rcond=None)[0]
    tss = float(((y - y.mean()) ** 2).sum())
    gcv = _gcv(rss, n, size, pen)
    gcv_null = _gcv(tss, n, 1, pen)
    glm = None
    if binomial:
        beta, dev, null = _irls(Bs, y)
        glm = {"coef": beta, "deviance": dev, "null_deviance": null, "df": n - len(cols)}
        coef = beta
    return MarsModel(sel_terms, coef, names, rss, gcv, 1 - rss / tss, 1 - gcv / gcv_null, n,
                     pen, len(terms), termination, cols, subsets, terms, glm)


def evimp(model: MarsModel) -> list[tuple[str, int, float, float]]:
    """earth's variable importance: (name, nsubsets, gcv, rss), criteria scaled to 100."""
    n, pen = model.n, model.penalty
    subs = sorted(model.subsets, key=lambda s: s[0])
    sel_size = len(model.selected)
    subs = [s for s in subs if s[0] <= sel_size]
    p = len(model.names)
    nsub = np.zeros(p, int)
    g = np.zeros(p)
    r = np.zeros(p)
    for i, (size, cols, rss) in enumerate(subs):
        used = set()
        for c in cols:
            used |= model.all_terms[c].vars()
        if i == 0:
            continue
        prev = subs[i - 1]
        drss = prev[2] - rss
        dgcv = _gcv(prev[2], n, prev[0], pen) - _gcv(rss, n, size, pen)
        for v in used:
            nsub[v] += 1
            r[v] += drss
            g[v] += dgcv
    out = []
    rmax, gmax = (r.max() if r.max() > 0 else 1), (g.max() if g.max() > 0 else 1)
    for v in np.argsort(-nsub * 1e9 - r, kind="mergesort"):
        if nsub[v] > 0:
            out.append((model.names[v], int(nsub[v]), 100 * g[v] / gmax, 100 * r[v] / rmax))
    return out


def mars_procedure(ds: Dataset, dep: str, predictors: list[str], degree: int = 1,
                   nk: int | None = None, thresh: float = 0.001, penalty: float | None = None,
                   minspan: int | None = None, endspan: int | None = None,
                   nprune: int | None = None, family: str = "gaussian",
                   plots: tuple = ("selection", "residuals", "partial"),
                   save: tuple = ()) -> MLResult:
    if not predictors:
        raise MLError("Select at least one predictor.")
    rows = complete_rows(ds, [dep] + list(predictors))
    if len(rows) < 5:
        raise MLError("Too few valid cases.")
    cols, names = [], []
    for nm in predictors:
        if measure_of(ds, nm) == "nominal" or ds.var(nm).type != "numeric":
            cat = categorical(ds, nm, rows)
            for j, lab in enumerate(cat.labels[1:], 1):
                cols.append((cat.codes == j).astype(float))
                names.append(f"{nm}{lab}")
        else:
            cols.append(raw_column(ds, nm).to_numpy(float)[rows])
            names.append(nm)
    X = np.column_stack(cols)
    binomial = family == "binomial"
    if binomial:
        cat = categorical(ds, dep, rows)
        if len(cat.levels) != 2:
            raise MLError("A binomial model needs a dependent variable with two values.")
        y = (cat.codes == 1).astype(float)
    else:
        y = raw_column(ds, dep).to_numpy(float)[rows]
    m = earth(X, y, names, degree, nk, thresh, minspan, endspan, penalty, nprune, binomial)
    section = Section("Multiple Adaptive Regression Splines")
    section.log(f"STATS EARTH DEPENDENT={dep} ENTER={' '.join(predictors)}\n"
                f"  /OPTIONS DEGREE={degree} PENALTY={m.penalty:g} THRESHOLD={thresh:g}"
                f"{' GLM=BINOMIAL' if binomial else ''}.\n"
                "* R-equivalent: earth(" + f"{dep} ~ {' + '.join(predictors)}, degree={degree}"
                + (", glm=list(family=binomial)" if binomial else "") + ").")
    section.table(Table("Coefficients", ["Basis Function", "Coefficient"],
                        [[t.label(names), fmt(c, 7)] for t, c in zip(m.terms, m.coef)],
                        footnotes=["h(x - t) = max(0, x - t); h(t - x) = max(0, t - x).",
                                   "Logistic regression (binomial GLM) coefficients."
                                   if binomial else
                                   "Least-squares coefficients of the selected terms."]))
    used = sorted({names[v] for t in m.terms for v in t.vars()})
    by_degree = np.bincount([t.degree for t in m.terms], minlength=degree + 1)
    rows_s = [["Selected Terms", f"{len(m.terms)} of {m.forward_terms}"],
              ["Selected Predictors", f"{len(used)} of {len(names)}"],
              ["Termination Condition", m.termination],
              ["Terms at Each Degree of Interaction",
               " ".join(str(int(v)) for v in by_degree) + (" (additive model)"
                                                            if by_degree[2:].sum() == 0 else "")],
              ["GCV", fmt(m.gcv, 4)], ["RSS", fmt(m.rss, 4)], ["GRSq", fmt(m.grsq, 4)],
              ["RSq", fmt(m.rsq, 4)], ["N", fmt_int(m.n)]]
    if binomial:
        rows_s += [["GLM Null Deviance", fmt(m.glm["null_deviance"], 4)],
                   ["GLM Deviance", fmt(m.glm["deviance"], 4)],
                   ["GLM Residual df", fmt_int(m.glm["df"])]]
    section.table(Table("Model Summary", ["", ""], rows_s,
                        footnotes=["GRSq = 1 - GCV/GCV(intercept only); GCV penalty = "
                                   f"{m.penalty:g} per knot."]))
    imp = evimp(m)
    if imp:
        section.table(Table("Variable Importance", ["Predictor", "nsubsets", "gcv", "rss"],
                            [[a, str(b), fmt(c, 1), fmt(d, 1)] for a, b, c, d in imp]))
    if "selection" in plots:
        subs = sorted(m.subsets, key=lambda s: s[0])
        tss = float(((y - y.mean()) ** 2).sum())
        sizes = [s[0] for s in subs]
        gcv_null = _gcv(tss, m.n, 1, m.penalty)
        fig = figure(5.4, 3.8)
        ax = fig.add_subplot(111)
        ax.plot(sizes, [1 - s[2] / tss for s in subs], label="RSq", color="#999")
        ax.plot(sizes, [1 - _gcv(s[2], m.n, s[0], m.penalty) / gcv_null for s in subs],
                label="GRSq", color="C3")
        ax.axvline(len(m.terms), ls="--", color="#666", lw=0.8)
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
        ax.legend(fontsize=8)
        style(ax, "Model Selection", "Number of terms", "GRSq / RSq")
        section.image("Model Selection", png(fig), width=460)
    fitted = m.predict(X, "response")
    if "residuals" in plots:
        fig = figure(5.0, 3.8)
        ax = fig.add_subplot(111)
        ax.scatter(fitted, y - fitted, s=12)
        ax.axhline(0, color="#999", lw=0.8)
        style(ax, "Residuals vs Fitted", "Fitted", "Residuals")
        section.image("Residuals vs Fitted", png(fig), width=440)
    if "partial" in plots and used:
        k = len(used)
        ncol = min(3, k)
        nrow = math.ceil(k / ncol)
        fig = figure(3.0 * ncol, 2.6 * nrow)
        med = np.median(X, 0)
        for i, nm in enumerate(used):
            v = names.index(nm)
            ax = fig.add_subplot(nrow, ncol, i + 1)
            grid = np.linspace(X[:, v].min(), X[:, v].max(), 100)
            G = np.tile(med, (100, 1))
            G[:, v] = grid
            ax.plot(grid, m.predict(G, "response"))
            style(ax, nm, nm, dep if i % ncol == 0 else "")
        section.image("Partial Dependence (other predictors at their medians)", png(fig),
                      width=min(900, 300 * ncol))
    saved = {}
    if "predicted" in save:
        saved["MARS_Predicted"] = ("MARS predicted value", full_series(ds, rows, fitted))
    if "residuals" in save:
        saved["MARS_Residual"] = ("MARS residual", full_series(ds, rows, y - fitted))
    return MLResult(section, saved, m)


def selftest() -> None:
    r = np.random.default_rng(7)
    x = np.sort(r.uniform(0, 10, 120))
    y = np.where(x > 5, 2 * (x - 5), 0) + r.normal(0, 0.2, 120)
    m = earth(x[:, None], y, ["x"])
    assert m.rsq > 0.95 and len(m.terms) >= 2

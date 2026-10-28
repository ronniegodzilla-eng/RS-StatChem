"""Analyze > Regression > Binary Logistic, Multinomial Logistic and Ordinal (SPSS LOGISTIC
REGRESSION, NOMREG and PLUM).

Binary logistic regression uses its own Newton-Raphson fit so the SPSS-specific pieces (Block 0
score tests, blocks of covariates with interaction terms (a*b), Forward/Backward Conditional, LR
and Wald selection, all seven contrasts, Hosmer-Lemeshow deciles of risk, residual and
influence saves, classification plot, casewise list) can be reproduced; estimates agree with
``statsmodels.Logit``. Multinomial uses ``statsmodels.MNLogit``
and ordinal uses ``statsmodels.OrderedModel``. As in SPSS NOMREG/PLUM, -2 log likelihood is the
full value (multinomial constant over covariate patterns included); pseudo R-squares use the
kernel.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special, stats

from rsstatchem.core.dataset import NUMERIC, Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, pct, sup, value_text

METHODS = {"enter": "Enter", "fstep_cond": "Forward: Conditional", "fstep_lr": "Forward: LR",
           "fstep_wald": "Forward: Wald", "bstep_cond": "Backward: Conditional",
           "bstep_lr": "Backward: LR", "bstep_wald": "Backward: Wald"}
CONTRASTS = ("indicator", "simple", "difference", "helmert", "repeated", "polynomial",
             "deviation")
LINKS = {"logit": "Logit", "probit": "Probit", "cloglog": "Complementary Log-log",
         "nloglog": "Negative Log-log", "cauchit": "Cauchit"}


@dataclass
class LogisticResult:
    section: Section
    saved: dict[str, tuple[str, pd.Series]] = field(default_factory=dict)
    models: list = field(default_factory=list)   # resample.ModelInfo (Bootstrap, MI pooling)


def _sort_key(v):
    return (isinstance(v, str), v)


def _pct(part: float, whole: float) -> str:
    return fmt(100 * part / whole, 1) if whole else ""


def _sig(p: float) -> str:
    return fmt_p(p) if p == p else ""


# ====================================================================== design helpers
@dataclass
class _Coding:
    name: str
    categories: list
    labels: list[str]
    matrix: np.ndarray          # categories x parameters
    contrast: str
    reference: str


@dataclass
class _Effect:
    name: str
    cols: list[str]
    categorical: bool


def contrast_coding(k: int, contrast: str, reference: str = "last") -> np.ndarray:
    """SPSS LOGISTIC REGRESSION parameter coding (categories x k-1) for each contrast.

    Indicator and deviation are coded directly; the others are the inverse of the full
    contrast matrix [1/k ... 1/k; L] without its first column, as SPSS documents."""
    m = np.zeros((k, k - 1))
    rows = list(range(k - 1)) if reference == "last" else list(range(1, k))
    if contrast in ("indicator", "deviation"):
        for j, r in enumerate(rows):
            m[r, j] = 1
        if contrast == "deviation":
            m[k - 1 if reference == "last" else 0, :] = -1
        return m
    ref = k - 1 if reference == "last" else 0
    L = np.zeros((k - 1, k))
    if contrast == "simple":
        for j, r in enumerate(rows):
            L[j, r] = 1
            L[j, ref] = -1
    elif contrast == "difference":
        for j in range(1, k):
            L[j - 1, j] = 1
            L[j - 1, :j] = -1 / j
    elif contrast == "helmert":
        for j in range(k - 1):
            L[j, j] = 1
            L[j, j + 1:] = -1 / (k - j - 1)
    elif contrast == "repeated":
        for j in range(k - 1):
            L[j, j], L[j, j + 1] = 1, -1
    elif contrast == "polynomial":
        q, _ = np.linalg.qr(np.vander(np.arange(1, k + 1, dtype=float), k, increasing=True))
        poly = q[:, 1:] * np.sign(q[-1, 1:])
        return poly
    else:
        raise ValueError(contrast)
    full = np.vstack([np.full(k, 1 / k), L])
    return np.linalg.inv(full)[:, 1:]


def _coding(ds: Dataset, name: str, values: pd.Series, contrast: str,
            reference: str) -> _Coding:
    var = ds.var(name)
    cats = sorted(values.dropna().unique(), key=_sort_key)
    m = contrast_coding(len(cats), contrast, reference) if len(cats) >= 2 else \
        np.zeros((len(cats), 0))
    return _Coding(name, cats, [value_text(var, c) for c in cats], m, contrast, reference)


def _encode(values: pd.Series, coding: _Coding) -> pd.DataFrame:
    index = {c: i for i, c in enumerate(coding.categories)}
    out = np.full((len(values), coding.matrix.shape[1]), np.nan)
    for r, v in enumerate(values.to_numpy(dtype=object)):
        i = index.get(v)
        if i is not None:
            out[r] = coding.matrix[i]
    cols = [f"{coding.name}({j + 1})" for j in range(coding.matrix.shape[1])]
    return pd.DataFrame(out, index=values.index, columns=cols)


def _raw(ds: Dataset, name: str, categorical: bool) -> pd.Series:
    if categorical or ds.var(name).type != NUMERIC:
        return ds.analysis_series(name)
    return numeric(ds, name)


def _term_parts(term: str) -> list[str]:
    return [p.strip() for p in term.split("*") if p.strip()]


def term_label(term: str) -> str:
    return " by ".join(_term_parts(term))


def _term_frame(term: str, raw: dict, codings: dict) -> tuple[pd.DataFrame, bool]:
    """Design columns of a main effect or an interaction (products of component columns)."""
    parts = _term_parts(term)
    frames = []
    for p in parts:
        if p in codings:
            frames.append(_encode(raw[p], codings[p]))
        else:
            frames.append(raw[p].astype(float).rename(p).to_frame())
    out = frames[0]
    for f in frames[1:]:
        cols = {}
        for a in out.columns:
            for b in f.columns:
                cols[f"{a} by {b}"] = out[a] * f[b]
        out = pd.DataFrame(cols, index=out.index)
    return out, any(p in codings for p in parts)


# ====================================================================== Newton-Raphson fit
@dataclass
class _Fit:
    cols: list[str]
    beta: np.ndarray
    cov: np.ndarray
    ll: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)

    @property
    def m2ll(self) -> float:
        return -2 * self.ll

    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))


def _loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def _newton(X: np.ndarray, y: np.ndarray, cols: list[str], max_iter: int = 20,
            bcon: float = 1e-3, start: np.ndarray | None = None) -> _Fit:
    """Newton-Raphson as SPSS: iterations are counted until parameter change < BCON."""
    p = X.shape[1]
    beta = np.zeros(p) if start is None else start.astype(float).copy()
    history, reported, converged = [], None, False
    for it in range(1, max(max_iter, 1) + 30):
        mu = special.expit(X @ beta)
        w = mu * (1 - mu)
        info = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(info, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info, X.T @ (y - mu), rcond=None)[0]
        beta = beta + step
        if reported is None:
            history.append((-2 * _loglik(X, y, beta), beta.copy()))
        change = float(np.max(np.abs(step))) if p else 0.0
        if reported is None and change < bcon:
            reported, converged = it, True
        if reported is None and it >= max_iter:
            reported = it
            break
        if change < 1e-10 or (reported is not None and it >= reported + 25):
            break
    mu = special.expit(X @ beta)
    info = (X * (mu * (1 - mu))[:, None]).T @ X
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    return _Fit(cols, beta, cov, _loglik(X, y, beta), reported or 1, converged, history)


def _score_test(X_full: np.ndarray, y: np.ndarray, mu: np.ndarray) -> float:
    u = X_full.T @ (y - mu)
    info = (X_full * (mu * (1 - mu))[:, None]).T @ X_full
    try:
        return float(u @ np.linalg.solve(info, u))
    except np.linalg.LinAlgError:
        return float(u @ np.linalg.pinv(info) @ u)


def _wald(beta: np.ndarray, cov: np.ndarray) -> float:
    try:
        return float(beta @ np.linalg.solve(cov, beta))
    except np.linalg.LinAlgError:
        return float("nan")


# ====================================================================== binary logistic
@dataclass
class _Step:
    effects: list[_Effect]
    fit: _Fit
    entered: list[str]
    removed: list[str]


def _pseudo_r2(ll0: float, ll1: float, n: int) -> tuple[float, float]:
    cs = 1 - math.exp(2 * (ll0 - ll1) / n)
    mx = 1 - math.exp(2 * ll0 / n)
    return cs, (cs / mx if mx > 0 else float("nan"))


def hosmer_lemeshow(y: np.ndarray, prob: np.ndarray, groups: int = 10):
    """Deciles of risk as SPSS forms them: tied probabilities stay in one group.

    Returns (chi-square, df, p, rows) with rows = (n, observed events, expected events).
    """
    order = np.argsort(prob, kind="mergesort")
    p_sorted, y_sorted = prob[order], y[order]
    n = len(y)
    keys = np.round(p_sorted, 10)
    uniq, start = np.unique(keys, return_index=True)
    bounds = list(start) + [n]
    assign = np.empty(n, dtype=int)
    for b in range(len(uniq)):
        lo, hi = bounds[b], bounds[b + 1]
        assign[lo:hi] = min(groups, max(1, math.ceil(groups * hi / n - 1e-9)))
    rows = []
    for g in range(1, groups + 1):
        mask = assign == g
        if mask.any():
            rows.append((int(mask.sum()), float(y_sorted[mask].sum()),
                         float(p_sorted[mask].sum())))
    chi = 0.0
    for m, o1, e1 in rows:
        o0, e0 = m - o1, m - e1
        chi += (o1 - e1) ** 2 / e1 if e1 > 0 else 0
        chi += (o0 - e0) ** 2 / e0 if e0 > 0 else 0
    df = len(rows) - 2
    p = float(stats.chi2.sf(chi, df)) if df > 0 else float("nan")
    return chi, df, p, rows


def logistic_diagnostics(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> dict:
    """SPSS LOGISTIC REGRESSION /SAVE statistics (leverage, residuals, Cook's, DfBeta)."""
    p = special.expit(X @ beta)
    w = p * (1 - p)
    info_inv = np.linalg.pinv((X * w[:, None]).T @ X)
    h = w * np.einsum("ij,jk,ik->i", X, info_inv, X)
    e = y - p
    one_h = np.clip(1 - h, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = e / np.sqrt(w)
        dev = np.sign(e) * np.sqrt(-2 * (y * np.log(np.clip(p, 1e-300, None))
                                         + (1 - y) * np.log(np.clip(1 - p, 1e-300, None))))
        sres = np.sign(e) * np.sqrt(dev ** 2 + h * z ** 2 / one_h)
        dfbeta = (X @ info_inv) * (e / one_h)[:, None]
    return {"prob": p, "lever": h, "resid": e, "lresid": e / w, "zresid": z, "dev": dev,
            "sresid": sres, "cook": z ** 2 * h / one_h ** 2, "dfbeta": dfbeta}


SAVE_LOGISTIC = {"prob": ("PRE", "Predicted probability"), "group": ("PGR", "Predicted group"),
                 "cook": ("COO", "Analog of Cook's influence statistics"),
                 "lever": ("LEV", "Leverage value"), "dfbeta": ("DFB", "DFBETA"),
                 "resid": ("RES", "Difference between observed and predicted probabilities"),
                 "lresid": ("LRE", "Logit residual"), "sresid": ("SRE", "Studentized residual"),
                 "zresid": ("ZRE", "Normalized residual"), "dev": ("DEV", "Deviance value")}
SAVE_KEYWORDS = {"prob": "PRED", "group": "PGROUP", "cook": "COOK", "lever": "LEVER",
                 "dfbeta": "DFBETA", "resid": "RESID", "lresid": "LRESID", "sresid": "SRESID",
                 "zresid": "ZRESID", "dev": "DEV"}
_METHOD_TEXT = {"enter": "Enter", "fstep_cond": "Forward Stepwise (Conditional)",
                "fstep_lr": "Forward Stepwise (Likelihood Ratio)",
                "fstep_wald": "Forward Stepwise (Wald)",
                "bstep_cond": "Backward Stepwise (Conditional)",
                "bstep_lr": "Backward Stepwise (Likelihood Ratio)",
                "bstep_wald": "Backward Stepwise (Wald)"}
_METHOD_KW = {"enter": "ENTER", "fstep_cond": "FSTEP(COND)", "fstep_lr": "FSTEP(LR)",
              "fstep_wald": "FSTEP(WALD)", "bstep_cond": "BSTEP(COND)",
              "bstep_lr": "BSTEP(LR)", "bstep_wald": "BSTEP(WALD)"}


def binary_logistic(ds: Dataset, dependent: str, covariates: list[str] | None = None,
                    categorical: dict[str, tuple[str, str]] | None = None,
                    method: str = "enter", cut: float = 0.5, p_enter: float = 0.05,
                    p_remove: float = 0.10, max_iter: int = 20, constant: bool = True,
                    hosmer: bool = True, ci: bool = True, level: float = 0.95,
                    iteration_history: bool = False, each_step: bool = True,
                    save=(), *, blocks: list[tuple[str, list[str]]] | None = None,
                    class_plot: bool = False, casewise: bool = False,
                    casewise_sd: float = 2.0, casewise_all: bool = False,
                    correlations: bool = False) -> LogisticResult:
    """``blocks`` = [(method, terms)], a term being a covariate or an interaction "a*b";
    without it one block (method, covariates) is used. ``categorical`` maps a variable to
    (contrast, reference) with contrast in :data:`CONTRASTS` and reference "last"|"first".
    ``save`` keys are those of :data:`SAVE_LOGISTIC`."""
    if blocks is None:
        blocks = [(method, list(covariates or []))]
    blocks = [(m, list(t)) for m, t in blocks if t]
    terms = list(dict.fromkeys(t for _, ts in blocks for t in ts))
    variables = list(dict.fromkeys(p for t in terms for p in _term_parts(t)))
    categorical = dict(categorical or {})
    for name in variables:
        if ds.var(name).type != NUMERIC and name not in categorical:
            categorical[name] = ("indicator", "last")
    categorical = {k: v for k, v in categorical.items() if k in variables}
    dvar = ds.var(dependent)
    section = Section("Logistic Regression")
    cat_sub = "".join(f"\n  /CONTRAST ({c})={categorical[c][0].capitalize()}"
                      + ("(1)" if categorical[c][1] == "first" else "")
                      for c in variables if c in categorical)
    prints = ["GOODFIT"] * hosmer + ([f"CI({level * 100:g})"] if ci else []) + \
        ["ITER(1)"] * iteration_history + ["CORR"] * correlations
    section.log(f"LOGISTIC REGRESSION VARIABLES {dependent}"
                + "".join(f"\n  /METHOD={_METHOD_KW[m]} " + " ".join(
                    t.replace("*", " BY ") if "*" in t else t for t in ts) for m, ts in blocks)
                + cat_sub
                + ("\n  /SAVE=" + " ".join(SAVE_KEYWORDS[s] for s in SAVE_LOGISTIC if s in save)
                   if save else "")
                + ("\n  /CLASSPLOT" if class_plot else "")
                + (("\n  /CASEWISE OUTLIER(" + f"{casewise_sd:g})" if not casewise_all else
                    "\n  /CASEWISE") if casewise else "")
                + "\n  /PRINT=" + " ".join(prints)
                + f"\n  /CRITERIA=PIN({p_enter:g}) POUT({p_remove:g}) ITERATE({max_iter})"
                  f" CUT({cut:g}).")
    result = LogisticResult(section)
    if not terms:
        section.note("Warning", "No covariates were specified.", style="warning")
        return result

    raw = {c: _raw(ds, c, c in categorical) for c in variables}
    ydata = ds.analysis_series(dependent)
    total = ds.n_cases
    frame = pd.DataFrame({"__y": ydata, **raw})
    complete = frame.notna().all(axis=1)
    frame = frame[complete]
    n = len(frame)
    missing = total - n

    section.table(Table("Case Processing Summary", ["Unweighted Casesᵃ", "", "N", "Percent"], [
        ["Selected Cases", "Included in Analysis", fmt_int(n), _pct(n, total)],
        ["", "Missing Cases", fmt_int(missing), _pct(missing, total)],
        ["", "Total", fmt_int(total), _pct(total, total)],
        ["Unselected Cases", "", "0", _pct(0, total)],
        ["Total", "", fmt_int(total), _pct(total, total)]], row_header_cols=2,
        footnotes=["a. If weight is in effect, see classification table for the total number "
                   "of cases."]))
    levels = sorted(frame["__y"].unique(), key=_sort_key)
    if len(levels) != 2:
        section.note("Warning", "The dependent variable has more than two non-missing values. "
                     "For logistic regression, the dependent variable must assume exactly two "
                     "values on the cases being processed." if len(levels) > 2 else
                     "The dependent variable must assume exactly two values on the cases "
                     "being processed.", style="warning")
        return result
    dlabels = [value_text(dvar, v) for v in levels]
    section.table(Table("Dependent Variable Encoding", ["Original Value", "Internal Value"],
                        [[dlabels[0], "0"], [dlabels[1], "1"]]))
    y = (frame["__y"] == levels[1]).to_numpy(float)

    codings: dict[str, _Coding] = {}
    for c in variables:
        if c in categorical:
            contrast, ref = categorical[c]
            coding = _coding(ds, c, frame[c], contrast, ref)
            if len(coding.categories) < 2:
                section.note("Warning", f"The categorical variable {c} has only one category "
                             "and was removed from the analysis.", style="warning")
                continue
            codings[c] = coding
    sub_raw = {c: frame[c] for c in variables}
    parts_df, effects_by_term = [], {}
    for t in terms:
        if any(p in categorical and p not in codings for p in _term_parts(t)):
            continue
        tf, is_cat = _term_frame(t, sub_raw, codings)
        parts_df.append(tf)
        effects_by_term[t] = _Effect(term_label(t), list(tf.columns), is_cat)
    if not effects_by_term:
        section.note("Warning", "No valid covariates.", style="warning")
        return result
    design = pd.concat(parts_df, axis=1)
    design = design.loc[:, ~design.columns.duplicated()]
    if constant:
        design["Constant"] = 1.0
    if codings:
        crow = []
        for c, coding in codings.items():
            counts = frame[c].value_counts()
            for i, (cat, lab) in enumerate(zip(coding.categories, coding.labels)):
                crow.append([c if i == 0 else "", lab, fmt_int(counts.get(cat, 0))]
                            + [fmt(v, 3) for v in coding.matrix[i]])
        width = max(m.matrix.shape[1] for m in codings.values())
        for r in crow:
            r += [""] * (3 + width - len(r))
        section.table(Table("Categorical Variables Codings", ["", "", "Frequency"]
                            + [("Parameter coding", f"({j + 1})") for j in range(width)],
                            crow, row_header_cols=2))

    def fit(effs: list[_Effect], start=None) -> _Fit:
        cols = [c for e in effs for c in e.cols] + (["Constant"] if constant else [])
        X = design[cols].to_numpy(float)
        if not cols:
            return _Fit([], np.zeros(0), np.zeros((0, 0)), float(len(y) * math.log(0.5)), 0,
                        True)
        return _newton(X, y, cols, max_iter, start=start)

    def probs(f: _Fit) -> np.ndarray:
        if not f.cols:
            return np.full(len(y), 0.5)
        return special.expit(design[f.cols].to_numpy(float) @ f.beta)

    def score_for(f: _Fit, cols: list[str]) -> tuple[float, int, float]:
        X = design[f.cols + cols].to_numpy(float)
        s = _score_test(X, y, probs(f))
        return s, len(cols), float(stats.chi2.sf(s, len(cols)))

    fit0 = fit([])
    ll0 = fit0.ll if constant else len(y) * math.log(0.5)
    p0 = probs(fit0)

    def classification(step_rows: list[tuple[str, np.ndarray]], title_cut: float):
        rows = []
        for label, prob in step_rows:
            pred = (prob > title_cut).astype(float)
            n00 = int(((y == 0) & (pred == 0)).sum())
            n01 = int(((y == 0) & (pred == 1)).sum())
            n10 = int(((y == 1) & (pred == 0)).sum())
            n11 = int(((y == 1) & (pred == 1)).sum())
            rows += [[label, dvar.title, dlabels[0], fmt_int(n00), fmt_int(n01),
                      _pct(n00, n00 + n01)],
                     ["", "", dlabels[1], fmt_int(n10), fmt_int(n11), _pct(n11, n10 + n11)],
                     ["", "Overall Percentage", "", "", "", _pct(n00 + n11, len(y))]]
        return rows

    # ---------------------------------------------------------------- Block 0
    section.note("Block 0: Beginning Block", "")
    if iteration_history and constant:
        hrows = [["Step 0", str(i + 1), fmt(m2, 3), fmt(b[-1], 3)]
                 for i, (m2, b) in enumerate(fit0.history)]
        for r in hrows[1:]:
            r[0] = ""
        section.table(Table("Iteration Historyᵃ ᵇ ᶜ", ["Iteration", "", "-2 Log likelihood",
                                                       ("Coefficients", "Constant")], hrows,
                            row_header_cols=2, footnotes=[
                                "a. Constant is included in the model.",
                                f"b. Initial -2 Log Likelihood: {fmt(fit0.m2ll, 3)}",
                                f"c. Estimation terminated at iteration number "
                                f"{fit0.iterations} because parameter estimates changed by "
                                "less than .001."]))
    ccols = ["Observed", "", "", ("Predicted", f"{dvar.title}: {dlabels[0]}"),
             ("Predicted", f"{dvar.title}: {dlabels[1]}"), ("Predicted", "Percentage Correct")]
    section.table(Table("Classification Tableᵃ ᵇ", ccols, classification([("Step 0", p0)], cut),
                        row_header_cols=3,
                        footnotes=(["a. Constant is included in the model."] if constant
                                   else []) + [f"b. The cut value is {fmt(cut, 3)}"]))
    vcols = ["", "", "B", "S.E.", "Wald", "df", "Sig.", "Exp(B)"]
    if constant:
        b, se = fit0.beta[-1], fit0.se()[-1]
        w = (b / se) ** 2
        section.table(Table("Variables in the Equation", vcols, [
            ["Step 0", "Constant", fmt(b), fmt(se), fmt(w), "1", fmt_p(stats.chi2.sf(w, 1)),
             fmt(math.exp(b))]], row_header_cols=2))

    def not_in_rows(f: _Fit, outside: list[_Effect], step_label: str) -> list[list[str]]:
        rows = []
        first = True
        for e in outside:
            s, df, p = score_for(f, e.cols)
            label = e.cols[0] if e.categorical and len(e.cols) == 1 else e.name
            rows.append([step_label if first else "", "Variables" if first else "", label,
                         fmt(s), fmt_int(df), fmt_p(p)])
            first = False
            if e.categorical and len(e.cols) > 1:
                for c in e.cols:
                    s1, d1, p1 = score_for(f, [c])
                    rows.append(["", "", c, fmt(s1), fmt_int(d1), fmt_p(p1)])
        all_cols = [c for e in outside for c in e.cols]
        s, df, p = score_for(f, all_cols)
        rows.append(["", "Overall Statistics", "", fmt(s), fmt_int(df), fmt_p(p)])
        return rows

    nc = ["", "", "", "Score", "df", "Sig."]
    all_effects = list(effects_by_term.values())
    section.table(Table("Variables not in the Equation", nc,
                        not_in_rows(fit0, all_effects, "Step 0"), row_header_cols=3))

    # ---------------------------------------------------------------- selection helpers
    def removal_stat(step_effects: list[_Effect], f: _Fit, e: _Effect, kind: str):
        idx = [f.cols.index(c) for c in e.cols]
        if kind == "lr":
            reduced = fit([x for x in step_effects if x is not e])
            change = reduced.m2ll - f.m2ll
            return change, len(e.cols), float(stats.chi2.sf(change, len(e.cols))), reduced.ll
        if kind == "cond":
            others = [i for i in range(len(f.cols)) if i not in idx]
            cov = f.cov
            b_red = f.beta[others] - cov[np.ix_(others, idx)] @ np.linalg.pinv(
                cov[np.ix_(idx, idx)]) @ f.beta[idx]
            X = design[[f.cols[i] for i in others]].to_numpy(float)
            ll_c = _loglik(X, y, b_red) if others else len(y) * math.log(0.5)
            change = -2 * (ll_c - f.ll)
            return change, len(e.cols), float(stats.chi2.sf(max(change, 0), len(e.cols))), ll_c
        w = _wald(f.beta[idx], f.cov[np.ix_(idx, idx)])
        return w, len(e.cols), float(stats.chi2.sf(w, len(e.cols))), None

    def run_block(meth: str, start_effs: list[_Effect], start_fit: _Fit,
                  block_effs: list[_Effect]) -> list[_Step]:
        steps: list[_Step] = []
        kind = meth.split("_")[1] if "_" in meth else "lr"
        if meth == "enter":
            new = [e for e in block_effs if e not in start_effs]
            effs = start_effs + new
            return [_Step(effs, fit(effs), [e.name for e in new], [])]
        current = list(start_effs)
        f = start_fit
        if meth.startswith("bstep"):
            new = [e for e in block_effs if e not in current]
            current = current + new
            f = fit(current)
            steps.append(_Step(list(current), f, [e.name for e in new], []))
        last_change = None
        for _ in range(4 * len(block_effs) + 4):
            changed = False
            order = ("remove", "enter") if meth.startswith("bstep") else ("enter", "remove")
            for action in order:
                if action == "enter":
                    best = None
                    for e in block_effs:
                        if e in current or last_change == ("remove", e.name):
                            continue
                        s, df, p = score_for(f, e.cols)
                        if best is None or p < best[1]:
                            best = (e, p)
                    if best is not None and best[1] <= p_enter:
                        current = current + [best[0]]
                        f = fit(current)
                        steps.append(_Step(list(current), f, [best[0].name], []))
                        last_change = ("enter", best[0].name)
                        changed = True
                        break
                else:
                    worst = None
                    for e in current:
                        if e not in block_effs or last_change == ("enter", e.name):
                            continue
                        _, _, p, _ = removal_stat(current, f, e, kind)
                        if worst is None or p > worst[1]:
                            worst = (e, p)
                    if worst is not None and worst[1] > p_remove:
                        current = [e for e in current if e is not worst[0]]
                        f = fit(current)
                        steps.append(_Step(list(current), f, [], [worst[0].name]))
                        last_change = ("remove", worst[0].name)
                        changed = True
                        break
            if not changed:
                break
        return steps

    # ---------------------------------------------------------------- blocks
    z = stats.norm.ppf(0.5 + level / 2)
    vcols_full = vcols + ([(f"{pct(level)} C.I.for EXP(B)", "Lower"),
                           (f"{pct(level)} C.I.for EXP(B)", "Upper")] if ci else [])
    cur_effs: list[_Effect] = []
    cur_fit = fit0
    last_fit, last_effs = None, None
    letters = iter(LETTERS_ALL)
    for bi, (meth, bterms) in enumerate(blocks, start=1):
        block_effs = [effects_by_term[t] for t in bterms if t in effects_by_term]
        method_text = _METHOD_TEXT[meth]
        section.note(f"Block {bi}: Method = {method_text}", "")
        steps = run_block(meth, cur_effs, cur_fit, block_effs)
        if not steps:
            section.note("", "No variables were entered into the equation (no score statistic "
                         f"was significant at the {fmt(p_enter)} level).", style="warning")
            continue
        block_ll = cur_fit.ll if cur_effs or constant else ll0
        shown = steps if each_step else steps[-1:]
        first_no = len(steps) - len(shown) + 1

        if iteration_history:
            hrows = []
            for k, s in enumerate(shown, start=first_no):
                for i, (m2, b) in enumerate(s.fit.history):
                    coefs = np.r_[b[-1:], b[:-1]] if constant else b
                    hrows.append([f"Step {k}" if i == 0 else "", str(i + 1), fmt(m2, 3)]
                                 + [fmt(v, 3) for v in coefs])
            if hrows:
                width = max(len(r) for r in hrows)
                names = max((s.fit.cols for s in shown), key=len)
                heads = (["Constant"] + names[:-1]) if constant else names
                for r in hrows:
                    r += [""] * (width - len(r))
                section.table(Table("Iteration Historyᵃ", ["Iteration", "", "-2 Log likelihood"]
                                    + [("Coefficients", h) for h in heads[:width - 3]], hrows,
                                    row_header_cols=2, footnotes=[
                                        f"a. Method: {method_text}",
                                        f"Initial -2 Log Likelihood: {fmt(fit0.m2ll, 3)}"]))

        orows, srows, hlrows, hlcont = [], [], [], []
        prev_ll = block_ll
        prev_df = len(cur_fit.cols) - (1 if constant else 0)
        block_df0 = prev_df
        for k, s in enumerate(steps, start=1):
            f = s.fit
            df_model = len(f.cols) - (1 if constant else 0)
            step_chi = 2 * (f.ll - prev_ll)
            step_df = df_model - prev_df
            block_chi = 2 * (f.ll - block_ll)
            block_df = df_model - block_df0
            model_chi = 2 * (f.ll - ll0)
            if k >= first_no:
                orows += [[f"Step {k}", "Step", fmt(step_chi), fmt_int(abs(step_df)),
                           _sig(float(stats.chi2.sf(abs(step_chi), abs(step_df))) if step_df
                                else float("nan"))],
                          ["", "Block", fmt(block_chi), fmt_int(abs(block_df)),
                           _sig(float(stats.chi2.sf(abs(block_chi), abs(block_df)))
                                if block_df else float("nan"))],
                          ["", "Model", fmt(model_chi), fmt_int(df_model),
                           _sig(float(stats.chi2.sf(model_chi, df_model)) if df_model
                                else float("nan"))]]
                cs, nk = _pseudo_r2(ll0, f.ll, len(y))
                srows.append([str(k), fmt(f.m2ll, 3) + sup("a"), fmt(cs), fmt(nk)])
                if hosmer:
                    chi, hdf, hp, groups = hosmer_lemeshow(y, probs(f))
                    hlrows.append([str(k), fmt(chi), fmt_int(hdf), _sig(hp)])
                    for g, (m, o1, e1) in enumerate(groups, start=1):
                        hlcont.append([f"Step {k}" if g == 1 else "", str(g), fmt_int(m - o1),
                                       fmt(m - e1, 3), fmt_int(o1), fmt(e1, 3), fmt_int(m)])
            prev_ll, prev_df = f.ll, df_model
        section.table(Table("Omnibus Tests of Model Coefficients", ["", "", "Chi-square", "df",
                                                                    "Sig."], orows,
                            row_header_cols=2,
                            footnotes=["A negative Chi-squares value indicates that the "
                                       "Chi-squares value has decreased from the previous step."]
                            if any(r[2].startswith("-") for r in orows) else []))
        last = steps[-1].fit
        term = (f"a. Estimation terminated at iteration number {last.iterations} because "
                "parameter estimates changed by less than .001." if last.converged else
                f"a. Estimation terminated at iteration number {last.iterations} because "
                "maximum iterations has been reached. Final solution cannot be found.")
        section.table(Table("Model Summary", ["Step", "-2 Log likelihood",
                                              "Cox & Snell R Square", "Nagelkerke R Square"],
                            srows, footnotes=[term]))
        if not last.converged:
            section.note("Warning", "The maximum number of iterations was reached; the "
                         "estimates may be infinite (complete or quasi-complete separation).",
                         style="warning")
        if hosmer:
            section.table(Table("Hosmer and Lemeshow Test", ["Step", "Chi-square", "df",
                                                             "Sig."], hlrows))
            g0, g1 = f"{dvar.title} = {dlabels[0]}", f"{dvar.title} = {dlabels[1]}"
            section.table(Table("Contingency Table for Hosmer and Lemeshow Test",
                                ["", "", (g0, "Observed"), (g0, "Expected"), (g1, "Observed"),
                                 (g1, "Expected"), "Total"], hlcont, row_header_cols=2))
        section.table(Table("Classification Tableᵃ", ccols, classification(
            [(f"Step {k}", probs(s.fit)) for k, s in enumerate(steps, start=1)
             if k >= first_no], cut), row_header_cols=3,
            footnotes=[f"a. The cut value is {fmt(cut, 3)}"]))

        vrows, notes = [], []
        for k, s in enumerate(steps, start=1):
            if k < first_no:
                continue
            f = s.fit
            se = f.se()
            letter = next(letters)
            what = s.entered and f"Variable(s) entered on step {k}: " + ", ".join(s.entered)
            what = what or f"Variable(s) removed on step {k}: " + ", ".join(s.removed)
            notes.append(f"{letter}. {what}.")
            first = True
            for e in s.effects:
                idx = [f.cols.index(c) for c in e.cols]
                if e.categorical and len(idx) > 1:
                    w = _wald(f.beta[idx], f.cov[np.ix_(idx, idx)])
                    vrows.append([f"Step {k}{sup(letter)}" if first else "", e.name, "", "",
                                  fmt(w), fmt_int(len(idx)),
                                  fmt_p(stats.chi2.sf(w, len(idx))), ""]
                                 + (["", ""] if ci else []))
                    first = False
                for i in idx:
                    vrows.append(_coef_row(f"Step {k}{sup(letter)}" if first else "",
                                           f.cols[i], f.beta[i], se[i], ci, z))
                    first = False
            if constant:
                vrows.append(_coef_row("", "Constant", f.beta[-1], se[-1], ci, z)[:8]
                             + (["", ""] if ci else []))
        section.table(Table("Variables in the Equation", vcols_full, vrows, row_header_cols=2,
                            footnotes=notes))

        if correlations:
            rrows = []
            for k, s in enumerate(steps, start=1):
                if k < first_no:
                    continue
                f = s.fit
                order = ([len(f.cols) - 1] if constant else []) + \
                    list(range(len(f.cols) - (1 if constant else 0)))
                sd = f.se()
                R = f.cov / np.outer(sd, sd)
                for a, i in enumerate(order):
                    rrows.append([f"Step {k}" if a == 0 else "", f.cols[i]]
                                 + [fmt(R[i, j]) for j in order])
            width = max(len(r) for r in rrows) if rrows else 2
            heads = ["Constant"] if constant else []
            heads += [c for c in steps[-1].fit.cols if c != "Constant"]
            for r in rrows:
                r += [""] * (width - len(r))
            section.table(Table("Correlation Matrix", ["", ""] + heads[:width - 2], rrows,
                                row_header_cols=2))

        if meth != "enter" and not meth.endswith("_wald"):
            kind = meth.split("_")[1]
            mrows = []
            for k, s in enumerate(steps, start=1):
                if k < first_no or not s.effects:
                    continue
                for j, e in enumerate([x for x in s.effects if x in block_effs]):
                    change, df, p, ll_red = removal_stat(s.effects, s.fit, e, kind)
                    mrows.append([f"Step {k}" if j == 0 else "", e.name, fmt(ll_red, 3),
                                  fmt(change, 3), fmt_int(df), fmt_p(p)])
            if mrows:
                section.table(Table("Model if Term Removed" + (sup("a") if kind == "cond"
                                                               else ""),
                                    ["", "Variable", "Model Log Likelihood",
                                     "Change in -2 Log Likelihood", "df",
                                     "Sig. of the Change"], mrows, row_header_cols=2,
                                    footnotes=["a. Based on conditional parameter estimates"]
                                    if kind == "cond" else []))
        if meth != "enter":
            nrows = []
            for k, s in enumerate(steps, start=1):
                outside = [e for e in all_effects if e not in s.effects]
                if k >= first_no and outside:
                    nrows += not_in_rows(s.fit, outside, f"Step {k}")
            if nrows:
                section.table(Table("Variables not in the Equation", nc, nrows,
                                    row_header_cols=3))
        cur_effs, cur_fit = steps[-1].effects, steps[-1].fit
        last_fit, last_effs = cur_fit, cur_effs

    if last_fit is None:
        return result
    Xa = design[last_fit.cols].to_numpy(float)
    dg = logistic_diagnostics(Xa, y, last_fit.beta)
    if last_fit.cols:
        from rsstatchem.core.stats import boot_models as bm
        step_label = f"Step {len(steps)}" if steps else "Step 1"
        info = bm.logit_model(y, Xa, [(step_label if i == 0 else "", c)
                                      for i, c in enumerate(last_fit.cols)],
                              last_fit.beta, max_iter, se=last_fit.se(),
                              rows=frame.index.to_numpy(), table="Variables in the Equation",
                              headers=["", ""], stat="wald", exp=True, stat_label="Wald")
        if any(m != "enter" for m, _ in blocks):
            info.note = ("Variable selection is not repeated in the bootstrap samples: the "
                         "final model is re-estimated with the selected variables.")
        result.models = [info]
    if class_plot:
        section.image("Observed Groups and Predicted Probabilities",
                      _class_plot(y, dg["prob"], dlabels, cut, dvar.title), width=520)
    if casewise:
        hit = np.ones(len(y), bool) if casewise_all else np.abs(dg["sresid"]) > casewise_sd
        crows = []
        idx_cases = frame.index.to_numpy() + 1
        for i in np.flatnonzero(hit):
            pg = 1 if dg["prob"][i] > cut else 0
            mis = "**" if pg != y[i] else ""
            crows.append([fmt_int(idx_cases[i]), "S", dlabels[int(y[i])] + mis,
                          fmt(dg["prob"][i]), dlabels[pg], fmt(dg["resid"][i]),
                          fmt(dg["zresid"][i], 2)])
        if crows:
            section.table(Table("Casewise Listᵇ", [
                "Case", "Selected Statusᵃ", (f"Observed", dvar.title), "Predicted",
                "Predicted Group", ("Temporary Variable", "Resid"),
                ("Temporary Variable", "ZResid")], crows, footnotes=[
                "a. S = Selected, U = Unselected cases, and ** = Misclassified cases.",
                "b. Cases with studentized residuals greater than "
                f"{fmt(casewise_sd, 3)} are listed." if not casewise_all else
                "b. All cases are listed."]))
        else:
            section.note("Casewise List", "The casewise plot is not produced because no "
                         "outliers were found.")

    # ---------------------------------------------------------------- save
    if save:
        parts = []
        full_raw = {c: _raw(ds, c, c in categorical) for c in variables}
        for e_term, e in effects_by_term.items():
            if e in last_effs:
                tf, _ = _term_frame(e_term, full_raw, codings)
                parts.append(tf)
        full = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ds.df.index)
        full = full.loc[:, ~full.columns.duplicated()]
        if constant:
            full["Constant"] = 1.0
        Xall = full[last_fit.cols].to_numpy(float)
        prob = pd.Series(special.expit(Xall @ last_fit.beta), index=ds.df.index)
        index = ds.df.index
        for key in [k for k in SAVE_LOGISTIC if k in save]:
            root, label = SAVE_LOGISTIC[key]
            if key == "prob":
                result.saved[root] = (label, prob)
            elif key == "group":
                codes = [v if isinstance(v, (int, float)) else np.nan for v in levels]
                grp = pd.Series(np.where(prob > cut, codes[1], codes[0]), index=index)
                grp[prob.isna()] = np.nan
                result.saved[root] = (label, grp.astype(float))
            elif key == "dfbeta":
                order = ([len(last_fit.cols) - 1] if constant else []) + \
                    list(range(len(last_fit.cols) - (1 if constant else 0)))
                for j, i in enumerate(order):
                    s = pd.Series(np.nan, index=index)
                    s[frame.index] = dg["dfbeta"][:, i]
                    result.saved[f"{root}{j}"] = (f"{label} for {last_fit.cols[i]}", s)
            else:
                s = pd.Series(np.nan, index=index)
                s[frame.index] = dg[key]
                result.saved[root] = (label, s)
    return result


LETTERS_ALL = "abcdefghijklmnopqrstuvwxyz"


def _class_plot(y, prob, dlabels, cut, title) -> bytes:
    """SPSS CLASSPLOT as a stacked histogram of predicted probabilities by observed group."""
    from rsstatchem.core.stats.roc import PALETTE, figure, png, style

    fig = figure(6.4, 3.8)
    ax = fig.add_subplot()
    bins = np.linspace(0, 1, 41)
    ax.hist([prob[y == 0], prob[y == 1]], bins=bins, stacked=True,
            color=[PALETTE[1], PALETTE[0]], edgecolor="white",
            label=[f"{dlabels[0][:1].upper() or '0'} = {dlabels[0]}",
                   f"{dlabels[1][:1].upper() or '1'} = {dlabels[1]}"])
    ax.axvline(cut, color="#333333", lw=1, ls="--")
    ax.set_xlim(0, 1)
    style(ax, "Observed Groups and Predicted Probabilities", "Predicted Probability",
          "Frequency")
    ax.legend(fontsize=8, frameon=False, title=title, title_fontsize=8)
    return png(fig)


def _coef_row(head: str, name: str, b: float, se: float, ci: bool, z: float) -> list[str]:
    w = (b / se) ** 2 if se > 0 else float("nan")
    row = [head, name, fmt(b), fmt(se), fmt(w), "1", _sig(float(stats.chi2.sf(w, 1))),
           fmt(_exp(b))]
    if ci:
        row += [fmt(_exp(b - z * se)), fmt(_exp(b + z * se))]
    return row


def _exp(x: float) -> float:
    return math.exp(x) if x < 700 else float("inf")


# ====================================================================== shared for NOMREG / PLUM
@dataclass
class _Data:
    y: np.ndarray                 # integer codes 0..J-1 in sorted category order
    levels: list
    labels: list[str]
    X: pd.DataFrame               # model columns (redundant levels dropped)
    terms: list[tuple[str, list[str], list[str]]]   # effect, columns, all-level labels
    factor_levels: dict[str, list]
    n_missing: int
    frame: pd.DataFrame
    spec: list[str] = field(default_factory=list)   # model terms ("a", "a*b")


def model_terms(factors: list[str], covariates: list[str], model: str = "main",
                terms: list[str] | None = None) -> list[str]:
    """Main effects (covariates then factors), full factorial (all factor interactions plus
    covariate main effects) or a custom list of terms such as ["x1", "g", "g*x1"]."""
    import itertools

    if model == "custom" and terms:
        return list(dict.fromkeys(terms))
    out = list(covariates) + list(factors)
    if model == "full":
        for size in range(2, len(factors) + 1):
            out += ["*".join(c) for c in itertools.combinations(factors, size)]
    return out


def _build(frame: pd.DataFrame, factors: list[str], terms: list[str], flev: dict):
    import itertools

    cols, out = {}, []
    for t in terms:
        parts = [p for p in t.split("*") if p]
        comps = []
        for p in parts:
            if p in factors:
                lv = flev[p]
                comps.append([(f"[{p}={format_code(v)}]",
                               (frame[p] == v).astype(float).where(frame[p].notna()),
                               i == len(lv) - 1) for i, v in enumerate(lv)])
            else:
                comps.append([(p, frame[p].astype(float), False)])
        all_names, keep = [], []
        for combo in itertools.product(*comps):
            name = " * ".join(c[0] for c in combo)
            all_names.append(name)
            if not any(c[2] for c in combo):
                s = combo[0][1]
                for c in combo[1:]:
                    s = s * c[1]
                cols[name] = s
                keep.append(name)
        out.append((" * ".join(parts), keep, all_names))
    X = pd.DataFrame(cols, index=frame.index) if cols else pd.DataFrame(index=frame.index)
    return X, out


def _prepare(ds: Dataset, dependent: str, factors: list[str], covariates: list[str],
             terms: list[str] | None = None, extra: list[str] = ()) -> _Data:
    used = list(dict.fromkeys(list(factors) + list(covariates) + list(extra)))
    frame = pd.DataFrame({"__y": ds.analysis_series(dependent),
                          **{f: ds.analysis_series(f) for f in factors},
                          **{c: numeric(ds, c) for c in used if c not in factors}})
    n_total = len(frame)
    frame = frame.dropna()
    levels = sorted(frame["__y"].unique(), key=_sort_key)
    dvar = ds.var(dependent)
    code = {v: i for i, v in enumerate(levels)}
    y = frame["__y"].map(code).to_numpy(int)
    flev = {f: sorted(frame[f].unique(), key=_sort_key) for f in factors}
    spec = terms if terms is not None else model_terms(factors, covariates)
    X, tout = _build(frame, list(factors), spec, flev)
    return _Data(y, levels, [value_text(dvar, v) for v in levels], X, tout, flev,
                 n_total - len(frame), frame, list(spec))


def _patterns(X: pd.DataFrame, y: np.ndarray, prob: np.ndarray, J: int):
    """Observed and expected counts per covariate pattern (SPSS subpopulations)."""
    key = _pattern_key(X, len(y))
    S = key.max() + 1
    obs = np.zeros((S, J))
    exp_ = np.zeros((S, J))
    np.add.at(obs, (key, y), 1)
    np.add.at(exp_, key, prob)
    return obs, exp_


def _pattern_key(X: pd.DataFrame, n: int) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(n, dtype=int)
    _, key = np.unique(np.round(X.to_numpy(float), 10), axis=0, return_inverse=True)
    return key.ravel()


def _multinomial_constant(obs: np.ndarray) -> float:
    return float(np.sum(special.gammaln(obs.sum(axis=1) + 1))
                 - np.sum(special.gammaln(obs + 1)))


def _gof_values(obs, exp_, n_params) -> tuple[float, float, int]:
    S, J = obs.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = float(np.nansum(np.where(exp_ > 0, (obs - exp_) ** 2 / exp_, 0)))
        dev = float(2 * np.nansum(np.where(obs > 0, obs * np.log(obs / exp_), 0)))
    return pearson, dev, S * (J - 1) - n_params


def _gof_rows(obs, exp_, n_params) -> tuple[list[list[str]], int, float]:
    pearson, dev, df = _gof_values(obs, exp_, n_params)
    rows = [["Pearson", fmt(pearson), fmt_int(df), _sig(stats.chi2.sf(pearson, df))
             if df > 0 else ""],
            ["Deviance", fmt(dev), fmt_int(df), _sig(stats.chi2.sf(dev, df)) if df > 0 else ""]]
    zero = int((obs == 0).sum())
    return rows, zero, 100 * zero / obs.size


def _case_summary(ds: Dataset, dependent: str, data: _Data, factors: list[str]) -> Table:
    n = len(data.y)
    rows = []
    dvar = ds.var(dependent)
    counts = np.bincount(data.y, minlength=len(data.levels))
    for i, lab in enumerate(data.labels):
        rows.append([dvar.title if i == 0 else "", lab, fmt_int(counts[i]),
                     _pct(counts[i], n) + "%"])
    for f in factors:
        fv = ds.var(f)
        vc = data.frame[f].value_counts()
        for i, v in enumerate(data.factor_levels[f]):
            rows.append([fv.title if i == 0 else "", value_text(fv, v), fmt_int(vc.get(v, 0)),
                         _pct(vc.get(v, 0), n) + "%"])
    total = n + data.n_missing
    rows += [["Valid", "", fmt_int(n), "100.0%"], ["Missing", "", fmt_int(data.n_missing), ""],
             ["Total", "", fmt_int(total), ""]]
    obs, _ = _patterns(data.X, data.y, np.zeros((n, len(data.levels))), len(data.levels))
    rows.append(["Subpopulation", "", fmt_int(obs.shape[0]) + sup("a"), ""])
    return Table("Case Processing Summary", ["", "", "N", "Marginal Percentage"], rows,
                 row_header_cols=2, footnotes=[
                     f"a. The dependent variable has only one value observed in "
                     f"{int(((obs > 0).sum(axis=1) == 1).sum())} "
                     f"({fmt(100 * ((obs > 0).sum(axis=1) == 1).mean(), 1)}%) subpopulations."])


def _null_ll(y: np.ndarray, J: int) -> float:
    counts = np.bincount(y, minlength=J).astype(float)
    counts = counts[counts > 0]
    return float(np.sum(counts * np.log(counts / counts.sum())))


def _pseudo_table(ll0: float, ll1: float, n: int) -> Table:
    cs, nk = _pseudo_r2(ll0, ll1, n)
    mf = 1 - ll1 / ll0 if ll0 else float("nan")
    return Table("Pseudo R-Square", ["", ""], [["Cox and Snell", fmt(cs)],
                                              ["Nagelkerke", fmt(nk)], ["McFadden", fmt(mf)]])


def _matrix_table(title: str, names: list[str], M: np.ndarray, heads: list[str] | None = None
                  ) -> Table:
    rows = [[nm] + [fmt(v) for v in M[i]] for i, nm in enumerate(names)]
    return Table(title, [""] + (heads or names), rows)


def _pattern_label(data: _Data, ds: Dataset, idx: int, names: list[str]) -> str:
    row = data.frame.iloc[idx]
    parts = []
    for n in names:
        var = ds.var(n)
        parts.append(f"{var.title} = {value_text(var, row[n])}")
    return ", ".join(parts)


# ====================================================================== multinomial (NOMREG)
def _mnlogit(y: np.ndarray, X: np.ndarray, J: int, max_iter: int = 100, tol: float = 1e-8):
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.MNLogit(y, X)
        try:
            res = model.fit(method="newton", maxiter=max_iter, disp=0, tol=tol)
        except np.linalg.LinAlgError:
            res = model.fit(method="bfgs", maxiter=max(2000, max_iter), disp=0)
    return res


STEP_METHODS = {"forward_entry": "Forward entry", "backward_elim": "Backward elimination",
                "forward_step": "Forward stepwise", "backward_step": "Backward stepwise"}


def multinomial(ds: Dataset, dependent: str, factors: list[str] = (),
                covariates: list[str] = (), reference: str = "last", level: float = 0.95,
                classification: bool = True, gof: bool = True, lr_tests: bool = True,
                save=(), *, model: str = "main", terms: list[str] | None = None,
                stepwise_terms: list[str] = (), step_method: str = "forward_entry",
                p_entry: float = 0.05, p_removal: float = 0.10, pseudo: bool = True,
                step_summary: bool = True, fitting_info: bool = True,
                info_criteria: bool = False, cell_probs: bool = False,
                monotonicity: bool = False, estimates: bool = True,
                correlations: bool = False, covariances: bool = False, max_iter: int = 100,
                p_converge: float = 1e-6, scale="none") -> LogisticResult:
    """``reference`` is "last", "first" or the code of a category (as text). ``model`` is
    main / full / custom (``terms``); ``stepwise_terms`` are entered or removed by
    ``step_method`` with likelihood-ratio tests. ``scale`` is none / pearson / deviance or a
    number (dispersion scaling of the covariance matrix and LR statistics). ``save`` keys:
    est (all response probabilities), pre, max (predicted category probability), acp."""
    factors, covariates = list(factors), list(covariates)
    dvar = ds.var(dependent)
    section = Section("Nominal Regression")
    forced = model_terms(factors, covariates, model, terms)
    stepwise_terms = [t for t in stepwise_terms if t not in forced]
    ref_kw = {"last": "LAST", "first": "FIRST"}.get(reference, reference)
    section.log(f"NOMREG {dependent} (BASE={ref_kw} ORDER=ASCENDING)"
                + (" BY " + " ".join(factors) if factors else "")
                + (" WITH " + " ".join(covariates) if covariates else "")
                + "\n  /CRITERIA CIN(" + f"{level * 100:g}" + f") DELTA(0) MXITER({max_iter}) "
                  f"MXSTEP(5) CHKSEP(20) LCONVERGE(0) PCONVERGE({p_converge:g}) "
                  "SINGULAR(0.00000001)"
                + "\n  /MODEL" + (" = " + " ".join(t.replace("*", " * ") for t in forced)
                                  if model != "main" else "")
                + (f" | {step_method.upper()} = "
                   + " ".join(t.replace("*", " * ") for t in stepwise_terms)
                   if stepwise_terms else "")
                + (f"\n  /SCALE={str(scale).upper()}" if scale != "none" else "")
                + "\n  /INTERCEPT=INCLUDE\n  /PRINT=" + " ".join(
                    ["CLASSTABLE"] * classification + ["FIT"] * gof + ["CELLPROB"] * cell_probs
                    + ["CORB"] * correlations + ["COVB"] * covariances
                    + ["PARAMETER"] * estimates + ["SUMMARY"] * pseudo + ["LRT"] * lr_tests
                    + ["CPS"] + ["STEP"] * step_summary + ["MFI"] * fitting_info
                    + ["IC"] * info_criteria) + ".")
    result = LogisticResult(section)
    data = _prepare(ds, dependent, factors, covariates, forced + stepwise_terms)
    J = len(data.levels)
    n = len(data.y)
    if J < 2 or n < 3:
        section.note("Warning", "The dependent variable must have at least two observed "
                     "categories.", style="warning")
        return result
    if reference == "last":
        ref = J - 1
    elif reference == "first":
        ref = 0
    else:
        texts = [format_code(v) for v in data.levels]
        ref = texts.index(str(reference)) if str(reference) in texts else J - 1
    order = [ref] + [j for j in range(J) if j != ref]
    remap = {old: new for new, old in enumerate(order)}
    yr = np.array([remap[v] for v in data.y])
    section.table(_case_summary(ds, dependent, data, factors))
    term_cols = {name: cols for name, cols, _ in data.terms}
    term_key = {t: name for t, (name, _, _) in zip(data.spec, data.terms)}
    tol = max(p_converge, 1e-10)

    def design(cols: list[str], intercept: bool = True) -> np.ndarray:
        X = data.X[cols].to_numpy(float) if cols else np.zeros((n, 0))
        return np.column_stack([np.ones(n), X]) if intercept else X

    def cols_of(tlist: list[str]) -> list[str]:
        return [c for t in tlist for c in term_cols[term_key[t]]]

    def llf(tlist: list[str]) -> float:
        cols = cols_of(tlist)
        return float(_mnlogit(yr, design(cols), J, max_iter, tol).llf)

    # ---------------------------------------------------------------- stepwise
    current = list(forced)
    step_rows = []
    obs0, _ = _patterns(data.X, data.y, np.zeros((n, J)), J)
    const = _multinomial_constant(obs0)
    if stepwise_terms:
        backward = step_method.startswith("backward")
        if backward:
            current += stepwise_terms
        ll_cur = llf(current)
        step_rows.append(["0", "Entered", "Intercept" + "".join(
            f", {term_key[t]}" for t in current), fmt(-2 * (ll_cur + const)), "", "", ""])
        both = step_method.endswith("_step")
        for k in range(1, 4 * len(stepwise_terms) + 4):
            changed = False
            if not backward or both:
                best = None
                for t in stepwise_terms:
                    if t in current:
                        continue
                    ll_new = llf(current + [t])
                    chi = 2 * (ll_new - ll_cur)
                    df = len(term_cols[term_key[t]]) * (J - 1)
                    p = float(stats.chi2.sf(chi, df))
                    if best is None or p < best[0]:
                        best = (p, t, ll_new, chi, df)
                if best and best[0] <= p_entry:
                    current.append(best[1])
                    ll_cur = best[2]
                    step_rows.append([str(k), "Entered", term_key[best[1]],
                                      fmt(-2 * (ll_cur + const)), fmt(best[3]),
                                      fmt_int(best[4]), _sig(best[0])])
                    changed = True
            if (backward or both) and not changed:
                worst = None
                for t in [x for x in current if x in stepwise_terms]:
                    ll_red = llf([x for x in current if x != t])
                    chi = 2 * (ll_cur - ll_red)
                    df = len(term_cols[term_key[t]]) * (J - 1)
                    p = float(stats.chi2.sf(max(chi, 0), df))
                    if worst is None or p > worst[0]:
                        worst = (p, t, ll_red, chi, df)
                if worst and worst[0] > p_removal:
                    current.remove(worst[1])
                    ll_cur = worst[2]
                    step_rows.append([str(k), "Removed", term_key[worst[1]],
                                      fmt(-2 * (ll_cur + const)), fmt(worst[3]),
                                      fmt_int(worst[4]), _sig(worst[0])])
                    changed = True
            if not changed:
                break
        if step_summary:
            section.table(Table("Step Summary", [
                "Model", "Action", "Effect(s)",
                ("Model Fitting Criteria", "-2 Log Likelihood"),
                ("Effect Selection Tests", "Chi-Squareᵃ"), ("Effect Selection Tests", "df"),
                ("Effect Selection Tests", "Sig.")], step_rows, footnotes=[
                f"Stepwise Method: {STEP_METHODS.get(step_method, step_method)}",
                "a. The chi-square for entry is based on the likelihood ratio test."
                if not backward else
                "a. The chi-square for removal is based on the likelihood ratio test."]))
    final_terms = current
    all_cols = cols_of(final_terms)
    model_terms_used = [(term_key[t],) + next((c, a) for nm, c, a in data.terms
                                              if nm == term_key[t]) for t in final_terms]

    res = _mnlogit(yr, design(all_cols), J, max_iter, tol)
    prob_r = np.asarray(res.predict(design(all_cols)))
    prob = prob_r[:, [order.index(j) for j in range(J)]]
    Xf = data.X[all_cols]
    obs, exp_ = _patterns(Xf, data.y, prob, J)
    const = _multinomial_constant(obs)
    ll1 = float(res.llf)
    ll0 = _null_ll(data.y, J)
    n_par = (1 + len(all_cols)) * (J - 1)
    pearson, dev, gdf = _gof_values(obs, exp_, n_par)
    if scale == "pearson":
        phi = pearson / gdf if gdf > 0 else 1.0
    elif scale == "deviance":
        phi = dev / gdf if gdf > 0 else 1.0
    elif isinstance(scale, (int, float)):
        phi = float(scale)
    else:
        phi = 1.0
    chi = 2 * (ll1 - ll0) / phi
    df = n_par - (J - 1)
    if fitting_info:
        m0, m1 = -2 * (ll0 + const), -2 * (ll1 + const)
        crit_cols = [("Model Fitting Criteria", "-2 Log Likelihood")]
        rows0, rows1 = [fmt(m0)], [fmt(m1)]
        if info_criteria:
            k0, k1 = J - 1, n_par
            crit_cols = [("Model Fitting Criteria", "AIC"), ("Model Fitting Criteria", "BIC")] \
                + crit_cols
            rows0 = [fmt(m0 + 2 * k0), fmt(m0 + k0 * math.log(n))] + rows0
            rows1 = [fmt(m1 + 2 * k1), fmt(m1 + k1 * math.log(n))] + rows1
        section.table(Table("Model Fitting Information", ["Model"] + crit_cols + [
            ("Likelihood Ratio Tests", "Chi-Square"), ("Likelihood Ratio Tests", "df"),
            ("Likelihood Ratio Tests", "Sig.")], [
            ["Intercept Only"] + rows0 + ["", "", ""],
            ["Final"] + rows1 + [fmt(chi), fmt_int(df),
                                 _sig(stats.chi2.sf(chi, df)) if df > 0 else ""]]))
    if gof:
        rows, zero, zp = _gof_rows(obs, exp_, n_par)
        section.table(Table("Goodness-of-Fit", ["", "Chi-Square", "df", "Sig."], rows))
        if zero:
            section.note("Warning", f"There are {zero} ({fmt(zp, 1)}%) cells (i.e., dependent "
                         "variable levels by subpopulations) with zero frequencies.",
                         style="warning")
    if phi != 1.0:
        kind = {"pearson": "Pearson", "deviance": "Deviance"}.get(scale, "user")
        section.note("Dispersion Scaling", f"Scale value = {fmt(phi, 3)} ({kind}); "
                     "covariances and likelihood-ratio statistics are scaled.")
    if pseudo:
        section.table(_pseudo_table(ll0, ll1, n))
    if monotonicity and J == 2:
        section.table(_monotonicity(data.y, prob[:, 1]))

    if lr_tests:
        lrows = []
        has_factor = any(set(t.split("*")) & set(factors) for t in final_terms)
        if has_factor:
            lrows.append(["Intercept", fmt(-2 * (ll1 + const)) + sup("b"), ".000", "0", ""])
        else:
            r0 = _mnlogit(yr, design(all_cols, intercept=False), J, max_iter, tol) \
                if all_cols else None
            ll_r = float(r0.llf) if r0 is not None else n * math.log(1 / J)
            c = 2 * (ll1 - ll_r) / phi
            lrows.append(["Intercept", fmt(-2 * (ll_r + const)), fmt(c), fmt_int(J - 1),
                          _sig(stats.chi2.sf(c, J - 1))])
        for name, cols, _ in model_terms_used:
            keep = [c for c in all_cols if c not in cols]
            r = _mnlogit(yr, design(keep), J, max_iter, tol)
            c = 2 * (ll1 - float(r.llf)) / phi
            d = len(cols) * (J - 1)
            lrows.append([name, fmt(-2 * (float(r.llf) + const)), fmt(c), fmt_int(d),
                          _sig(stats.chi2.sf(max(c, 0), d))])
        section.table(Table("Likelihood Ratio Tests", [
            "Effect", ("Model Fitting Criteria", "-2 Log Likelihood of Reduced Model"),
            ("Likelihood Ratio Tests", "Chi-Square"), ("Likelihood Ratio Tests", "df"),
            ("Likelihood Ratio Tests", "Sig.")], lrows, footnotes=[
            "The chi-square statistic is the difference in -2 log-likelihoods between the final "
            "model and a reduced model. The reduced model is formed by omitting an effect from "
            "the final model. The null hypothesis is that all parameters of that effect are 0."]
            + (["b. This reduced model is equivalent to the final model because omitting the "
                "effect does not increase the degrees of freedom."] if has_factor else [])))

    params = np.asarray(res.params)
    cov_all = np.asarray(res.cov_params()) * phi
    bse = np.sqrt(np.clip(np.diag(cov_all), 0, None)).reshape(params.shape, order="F")
    from rsstatchem.core.stats import boot_models as bm
    pn = ["Intercept"] + all_cols
    info = bm.mnlogit_model(yr, design(all_cols), J,
                            [(data.labels[j] if i == 0 else "", nm)
                             for j in order[1:] for i, nm in enumerate(pn)], params,
                            se=bse.T.ravel(), rows=data.frame.index.to_numpy(),
                            table="Parameter Estimates", headers=[dvar.title, ""],
                            stat="wald", exp=True, stat_label="Wald")
    if stepwise_terms:
        info.note = ("Term selection is not repeated in the bootstrap samples: the final "
                     "model is re-estimated with the selected terms.")
    result.models = [info]
    z = stats.norm.ppf(0.5 + level / 2)
    ci_g = f"{pct(level)} Confidence Interval for Exp(B)"
    if estimates:
        prow = []
        for k, j in enumerate(order[1:]):
            b_col, s_col = params[:, k], bse[:, k]
            head = data.labels[j] + sup("a") if k == 0 else data.labels[j]
            prow.append(_mn_row(head, "Intercept", b_col[0], s_col[0], z, exp=False))
            for name, cols, all_names in model_terms_used:
                for nm in all_names:
                    if nm in cols:
                        i = 1 + all_cols.index(nm)
                        prow.append(_mn_row("", nm, b_col[i], s_col[i], z))
                    else:
                        prow.append(["", nm, "0" + sup("b"), "", "", "0", "", "", "", ""])
        section.table(Table("Parameter Estimates", [
            f"{dvar.title}", "", "B", "Std. Error", "Wald", "df", "Sig.", "Exp(B)",
            (ci_g, "Lower Bound"), (ci_g, "Upper Bound")], prow, row_header_cols=2,
            footnotes=[f"a. The reference category is: {data.labels[ref]}."]
            + (["b. This parameter is set to zero because it is redundant."]
               if any(len(c) < len(a) for _, c, a in model_terms_used) else [])))
    if correlations or covariances:
        names = []
        for k, j in enumerate(order[1:]):
            names += [f"{data.labels[j]}: {nm}" for nm in ["Intercept"] + all_cols]
        if covariances:
            section.table(_matrix_table("Asymptotic Covariance Matrix", names, cov_all))
        if correlations:
            sd = np.sqrt(np.clip(np.diag(cov_all), 1e-300, None))
            section.table(_matrix_table("Asymptotic Correlation Matrix", names,
                                        cov_all / np.outer(sd, sd)))
    if cell_probs:
        section.table(_cell_table(ds, data, prob, Xf, factors + covariates,
                                  "Observed and Predicted Frequencies", nomreg=True))
    if classification:
        section.table(_class_table(data, prob))
    if save:
        _save_category_probs(ds, data, prob, save, result,
                             lambda X: np.asarray(res.predict(
                                 np.column_stack([np.ones(len(X)), X])))[
                                 :, [order.index(j) for j in range(J)]],
                             final_terms, all_cols, dependent)
    return result


def _monotonicity(y: np.ndarray, p: np.ndarray) -> Table:
    pos, neg = p[y == 1], p[y == 0]
    conc = float(np.sum(pos[:, None] > neg[None, :]))
    disc = float(np.sum(pos[:, None] < neg[None, :]))
    ties = float(pos.size * neg.size - conc - disc)
    n = len(y)
    pairs = n * (n - 1) / 2
    total = conc + disc + ties
    return Table("Monotonicity Measures", ["", "Number", "Percent"], [
        ["Concordant Pairs", fmt_int(conc), _pct(conc, total) + "%"],
        ["Discordant Pairs", fmt_int(disc), _pct(disc, total) + "%"],
        ["Tied Pairs", fmt_int(ties), _pct(ties, total) + "%"],
        ["Total", fmt_int(total), "100.0%"],
        ["Somers' D", fmt((conc - disc) / total), ""],
        ["Goodman-Kruskal Gamma", fmt((conc - disc) / (conc + disc)) if conc + disc else "",
         ""],
        ["Kendall's Tau-a", fmt((conc - disc) / pairs), ""],
        ["Concordance Index C", fmt((conc + 0.5 * ties) / total), ""]])


def _cell_table(ds, data, prob, X, names, title, nomreg=False) -> Table:
    key = _pattern_key(X, len(data.y))
    J = len(data.levels)
    rows = []
    for s in range(key.max() + 1):
        idx = np.flatnonzero(key == s)
        label = _pattern_label(data, ds, idx[0], names) if names else "All cases"
        o = np.bincount(data.y[idx], minlength=J).astype(float)
        e = prob[idx].sum(axis=0)
        m = len(idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = (o - e) / np.sqrt(e * (1 - e / m))
        if nomreg:
            for j in range(J):
                rows.append([label if j == 0 else "", data.labels[j], fmt_int(o[j]), fmt(e[j]),
                             fmt(pr[j]), _pct(o[j], m) + "%", _pct(e[j], m) + "%"])
        else:
            rows.append([label, "Observed"] + [fmt_int(v) for v in o])
            rows.append(["", "Expected"] + [fmt(v) for v in e])
            rows.append(["", "Pearson Residual"] + [fmt(v) for v in pr])
    if nomreg:
        return Table(title, ["", "", ("Frequency", "Observed"), ("Frequency", "Predicted"),
                             ("Frequency", "Pearson Residual"),
                             ("Percentage", "Observed"), ("Percentage", "Predicted")], rows,
                     row_header_cols=2)
    return Table("Cell Information", ["", ""] + [("Frequency", lab) for lab in data.labels],
                 rows, row_header_cols=2)


def _mn_row(head, name, b, se, z, exp=True) -> list[str]:
    w = (b / se) ** 2 if se > 0 else float("nan")
    row = [head, name, fmt(b), fmt(se), fmt(w), "1", _sig(float(stats.chi2.sf(w, 1)))]
    row += [fmt(_exp(b)), fmt(_exp(b - z * se)), fmt(_exp(b + z * se))] if exp else ["", "", ""]
    return row


def _class_table(data: _Data, prob: np.ndarray) -> Table:
    J = len(data.levels)
    pred = prob.argmax(axis=1)
    n = len(data.y)
    rows = []
    for j in range(J):
        counts = [int(((data.y == j) & (pred == k)).sum()) for k in range(J)]
        tot = sum(counts)
        rows.append([data.labels[j]] + [fmt_int(c) for c in counts]
                    + [_pct(counts[j], tot) + "%"])
    marg = [int((pred == k).sum()) for k in range(J)]
    rows.append(["Overall Percentage"] + [_pct(m, n) + "%" for m in marg]
                + [_pct(int((pred == data.y).sum()), n) + "%"])
    return Table("Classification", ["Observed"] + [("Predicted", lab) for lab in data.labels]
                 + [("Predicted", "Percent Correct")], rows)


def _save_category_probs(ds, data, prob, save, result, predict, terms, cols,
                         dependent) -> None:
    factors = list(data.factor_levels)
    used = list(dict.fromkeys(p for t in terms for p in t.split("*")))
    frame = pd.DataFrame({u: (ds.analysis_series(u) if u in factors else numeric(ds, u))
                          for u in used}, index=ds.df.index)
    for f in factors:
        if f in frame:
            frame.loc[~frame[f].isin(data.factor_levels[f]), f] = np.nan
    X, _ = _build(frame, factors, terms, data.factor_levels)
    X = X[cols] if cols else X
    ok = X.notna().all(axis=1).to_numpy() & frame.notna().all(axis=1).to_numpy()
    J = len(data.levels)
    full = np.full((ds.n_cases, J), np.nan)
    if ok.any():
        full[ok] = predict(X.to_numpy(float)[ok])
    index = ds.df.index
    if "est" in save or "pcp" in save:
        for j in range(J):
            result.saved[f"EST{j + 1}"] = (f"Estimated Cell Probability for Response Category: "
                                           f"{data.labels[j]}", pd.Series(full[:, j],
                                                                          index=index))
    if "pre" in save or "group" in save:
        codes = [v if isinstance(v, (int, float)) else np.nan for v in data.levels]
        cat = np.full(ds.n_cases, np.nan)
        cat[ok] = np.asarray(codes, float)[np.nanargmax(full[ok], axis=1)]
        result.saved["PRE"] = ("Predicted Response Category", pd.Series(cat, index=index))
    if "max" in save or "pcp" in save:
        result.saved["PCP"] = ("Estimated Response Probability for the Predicted Category",
                               pd.Series(np.nanmax(full, axis=1) if ok.any() else full[:, 0],
                                         index=index))
    if "acp" in save:
        ys = ds.analysis_series(dependent)
        code = {v: i for i, v in enumerate(data.levels)}
        acp = np.full(ds.n_cases, np.nan)
        for r, v in enumerate(ys.to_numpy(dtype=object)):
            if ok[r] and v in code:
                acp[r] = full[r, code[v]]
        result.saved["ACP"] = ("Estimated Response Probability for the Actual Category",
                               pd.Series(acp, index=index))


# ====================================================================== ordinal (PLUM)
def _distr(link: str):
    return {"logit": stats.logistic, "probit": stats.norm, "cloglog": stats.gumbel_l,
            "nloglog": stats.gumbel_r, "cauchit": stats.cauchy}[link]


def _ordinal_fit(y: np.ndarray, X: np.ndarray, J: int, link: str):
    """Returns (thresholds, beta, cov over [thresholds, beta], loglik, converged)."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    k = X.shape[1]
    if k == 0:
        counts = np.bincount(y, minlength=J)
        cum = np.cumsum(counts)[:-1] / counts.sum()
        th = _distr(link).ppf(cum)
        n = len(y)
        f = _distr(link).pdf(th)
        # asymptotic covariance of sample cumulative proportions, delta method
        cov_p = np.minimum.outer(cum, cum) * (1 - np.maximum.outer(cum, cum)) / n
        jac = np.diag(1 / f)
        return th, np.zeros(0), jac @ cov_p @ jac, _null_ll(y, J), True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        distr = link if link in ("logit", "probit") else _distr(link)
        model = OrderedModel(y, X, distr=distr)
        res = model.fit(method="bfgs", maxiter=5000, disp=False, gtol=1e-8)
        try:
            res2 = model.fit(start_params=res.params, method="newton", maxiter=100,
                             disp=False)
            if np.isfinite(res2.llf) and res2.llf >= res.llf - 1e-9:
                res = res2
        except (np.linalg.LinAlgError, ValueError):
            pass
    params = np.asarray(res.params)
    beta, raw = params[:k], params[k:]
    th = np.r_[raw[0], raw[0] + np.cumsum(np.exp(raw[1:]))]
    jac = np.zeros((J - 1 + k, J - 1 + k))
    for i in range(J - 1):
        jac[i, k] = 1
        for m in range(1, i + 1):
            jac[i, k + m] = math.exp(raw[m])
    for i in range(k):
        jac[J - 1 + i, i] = 1
    cov = jac @ np.asarray(res.cov_params()) @ jac.T
    fisher = _plum_fisher_cov(th, beta, X, link)
    if fisher is not None and np.all(np.isfinite(fisher)):
        cov = fisher
    conv = bool(res.mle_retvals.get("converged", True))
    return th, beta, cov, float(res.llf), conv


def _plum_fisher_cov(th, beta, X, link, tau=None, Z=None, w=None) -> np.ndarray | None:
    """Inverse expected (Fisher) information of a cumulative link model over
    [thresholds, location, scale]. SPSS PLUM prints standard errors from this matrix
    (IBM SPSS Statistics Algorithms, PLUM: Fisher scoring covariance); the observed-Hessian
    covariance differs slightly (e.g. UCLA ologit: SE(pared) .268 Fisher vs .266 observed)."""
    d = _distr(link)
    n = X.shape[0]
    k = X.shape[1]
    J1 = len(th)
    m = Z.shape[1] if Z is not None and tau is not None and len(tau) else 0
    eta = X @ beta if k else np.zeros(n)
    s = np.exp(Z @ tau) if m else np.ones(n)
    z = (th[None, :] - eta[:, None]) / s[:, None]                     # n x (J-1)
    f = d.pdf(z)
    F = d.cdf(z)
    zero = np.zeros((n, 1))
    Fc = np.hstack([zero, F, zero + 1])
    fc = np.hstack([zero, f, zero])
    zc = np.hstack([zero, np.where(np.isfinite(z), z, 0.0), zero])
    pi = np.clip(np.diff(Fc, axis=1), 1e-300, None)                   # n x J
    wts = np.ones(n) if w is None else np.asarray(w, float)
    npar = J1 + k + m
    info = np.zeros((npar, npar))
    for c in range(J1 + 1):
        g = np.zeros((n, npar))
        if c < J1:
            g[:, c] += f[:, c] / s
        if c >= 1:
            g[:, c - 1] -= f[:, c - 1] / s
        dfc = (fc[:, c + 1] - fc[:, c]) / s
        if k:
            g[:, J1:J1 + k] = -X * dfc[:, None]
        if m:
            g[:, J1 + k:] = -Z * (fc[:, c + 1] * zc[:, c + 1] - fc[:, c] * zc[:, c])[:, None]
        info += (g * (wts / pi[:, c])[:, None]).T @ g
    try:
        return np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None


def plum_mle(y: np.ndarray, X: np.ndarray, Z: np.ndarray, J: int, link: str,
             max_iter: int = 100, start: tuple | None = None):
    """Cumulative link model with a scale component (SPSS PLUM /SCALE):
    P(Y <= j) = F((theta_j - x'beta) / exp(z'tau)). Returns (theta, beta, tau, cov, loglik,
    converged, history)."""
    from scipy import optimize
    from statsmodels.tools.numdiff import approx_hess3

    k, m = X.shape[1], Z.shape[1]
    d = _distr(link)
    counts = np.bincount(y, minlength=J)
    th0 = d.ppf(np.clip(np.cumsum(counts)[:-1] / counts.sum(), 1e-4, 1 - 1e-4))
    if start is not None:
        th0, b0 = start
    else:
        b0 = np.zeros(k)

    def unpack_raw(par):
        th = np.r_[par[0], par[0] + np.cumsum(np.exp(par[1:J - 1]))]
        return th, par[J - 1:J - 1 + k], par[J - 1 + k:]

    def nll_nat(th, beta, tau):
        eta = X @ beta if k else np.zeros(len(y))
        s = np.exp(Z @ tau) if m else np.ones(len(y))
        cdf = d.cdf((th[None, :] - eta[:, None]) / s[:, None])
        cdf = np.column_stack([np.zeros(len(y)), cdf, np.ones(len(y))])
        p = np.diff(cdf, axis=1)[np.arange(len(y)), y]
        return -float(np.sum(np.log(np.clip(p, 1e-300, None))))

    history = []

    def nll_raw(par):
        return nll_nat(*unpack_raw(par))

    gaps = np.log(np.clip(np.diff(th0), 1e-3, None))
    x0 = np.r_[th0[0], gaps, b0, np.zeros(m)]
    history.append((2 * nll_raw(x0), unpack_raw(x0)))
    res = optimize.minimize(nll_raw, x0, method="BFGS",
                            callback=lambda xk: history.append((2 * nll_raw(xk),
                                                                unpack_raw(xk))),
                            options={"maxiter": max(200, 50 * max_iter), "gtol": 1e-7})
    th, beta, tau = unpack_raw(res.x)
    nat = np.r_[th, beta, tau]

    def nll_flat(v):
        return nll_nat(v[:J - 1], v[J - 1:J - 1 + k], v[J - 1 + k:])

    cov = _plum_fisher_cov(th, beta, X, link, tau, Z)
    if cov is None or not np.all(np.isfinite(cov)):
        H = approx_hess3(nat, nll_flat)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
    return th, beta, tau, cov, -float(res.fun), bool(res.success), history


def _ordinal_probs(th, beta, X, link, tau=None, Z=None) -> np.ndarray:
    eta = X @ beta if len(beta) else np.zeros(X.shape[0])
    s = np.exp(Z @ tau) if tau is not None and len(tau) else np.ones(X.shape[0])
    cdf = _distr(link).cdf((th[None, :] - eta[:, None]) / s[:, None])
    cdf = np.column_stack([np.zeros(len(eta)), cdf, np.ones(len(eta))])
    return np.diff(cdf, axis=1)


def _general_ordinal_ll(y, X, J, link, th0, beta0):
    """Non-parallel cumulative model (Test of Parallel Lines): separate slopes per split."""
    from scipy import optimize

    k = X.shape[1]
    d = _distr(link)
    start = np.r_[th0, np.tile(beta0, J - 1)]

    def nll(par):
        th = par[:J - 1]
        B = par[J - 1:].reshape(J - 1, k)
        cdf = d.cdf(th[None, :] - X @ B.T)
        cdf = np.column_stack([np.zeros(len(y)), cdf, np.ones(len(y))])
        p = np.diff(cdf, axis=1)[np.arange(len(y)), y]
        return -float(np.sum(np.log(np.clip(p, 1e-300, None))))

    best = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-7,
                                                                 "maxiter": 20000})
    alt = optimize.minimize(nll, best.x, method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
    return -min(best.fun, alt.fun)


def ordinal(ds: Dataset, dependent: str, factors: list[str] = (), covariates: list[str] = (),
            link: str = "logit", level: float = 0.95, gof: bool = True,
            parallel: bool = False, classification: bool = False, save=(), *,
            location: list[str] | None = None, scale_terms: list[str] = (),
            summary: bool = True, estimates: bool = True, correlations: bool = False,
            covariances: bool = False, cell_info: bool = False,
            iteration_history: bool = False, max_iter: int = 100) -> LogisticResult:
    """``location`` = custom location terms (main effects by default, interactions "a*b"
    allowed); ``scale_terms`` = terms of the scale component (heteroscedastic model)."""
    factors, covariates = list(factors), list(covariates)
    dvar = ds.var(dependent)
    section = Section("PLUM - Ordinal Regression")
    loc_terms = model_terms(factors, covariates, "custom" if location else "main", location)
    scale_terms = list(scale_terms)
    section.log(f"PLUM {dependent}" + (" BY " + " ".join(factors) if factors else "")
                + (" WITH " + " ".join(covariates) if covariates else "")
                + f"\n  /CRITERIA=CIN({level * 100:g}) DELTA(0) LCONVERGE(0) MXITER({max_iter})"
                  " MXSTEP(5) PCONVERGE(1.0E-6) SINGULAR(1.0E-8)"
                + f"\n  /LINK={link.upper()}"
                + ("\n  /LOCATION=" + " ".join(t.replace("*", " * ") for t in loc_terms)
                   if location else "")
                + ("\n  /SCALE=" + " ".join(t.replace("*", " * ") for t in scale_terms)
                   if scale_terms else "")
                + "\n  /PRINT=" + " ".join(
                    ["FIT"] * gof + ["PARAMETER"] * estimates + ["SUMMARY"] * summary
                    + ["TPARALLEL"] * parallel + ["CORB"] * correlations
                    + ["COVB"] * covariances + ["CELLINFO"] * cell_info
                    + ["HISTORY(1)"] * iteration_history) + ".")
    result = LogisticResult(section)
    data = _prepare(ds, dependent, factors, covariates, loc_terms)
    J = len(data.levels)
    n = len(data.y)
    if J < 2:
        section.note("Warning", "The dependent variable must have at least two observed "
                     "categories.", style="warning")
        return result
    section.table(_case_summary(ds, dependent, data, factors))
    X = data.X.to_numpy(float)
    k = X.shape[1]
    Zdf, zterms = _build(data.frame, factors, scale_terms, data.factor_levels) if scale_terms \
        else (pd.DataFrame(index=data.frame.index), [])
    Z = Zdf.to_numpy(float)
    m = Z.shape[1]
    tau = np.zeros(0)
    history = []
    if m or iteration_history:
        th0, b0, *_ = _ordinal_fit(data.y, X, J, link) if k else (None, None)
        th, beta, tau, cov, ll1, conv, history = plum_mle(
            data.y, X, Z, J, link, max_iter, None if th0 is None else (th0, b0))
    else:
        th, beta, cov, ll1, conv = _ordinal_fit(data.y, X, J, link)
    ll0 = _null_ll(data.y, J)
    prob = _ordinal_probs(th, beta, X, link, tau, Z)
    pat = pd.concat([data.X, Zdf], axis=1)
    pat = pat.loc[:, ~pat.columns.duplicated()]
    obs, exp_ = _patterns(pat, data.y, prob, J)
    const = _multinomial_constant(obs)
    chi = 2 * (ll1 - ll0)
    if iteration_history and history:
        hrows = []
        for i, (m2, (t_, b_, s_)) in enumerate(history):
            hrows.append([str(i), fmt(m2 + 2 * (-const), 3), "0"]
                         + [fmt(v, 3) for v in np.r_[t_, b_, s_]])
        heads = [f"Threshold {j + 1}" for j in range(J - 1)] + list(data.X.columns) + \
            [f"Scale: {c}" for c in Zdf.columns]
        section.table(Table("Iteration History", ["Iteration", "-2 Log Likelihood",
                                                  "Step-halving"] + heads, hrows,
                            footnotes=[f"Link function: {LINKS[link]}.",
                                       "Iterations of the quasi-Newton (BFGS) optimiser."]))
    section.table(Table("Model Fitting Information", [
        "Model", "-2 Log Likelihood", "Chi-Square", "df", "Sig."], [
        ["Intercept Only", fmt(-2 * (ll0 + const)), "", "", ""],
        ["Final", fmt(-2 * (ll1 + const)), fmt(chi), fmt_int(k + m),
         _sig(stats.chi2.sf(chi, k + m)) if k + m else ""]],
        footnotes=[f"Link function: {LINKS[link]}."]))
    if not conv:
        section.note("Warning", "The estimation did not converge; results may not be "
                     "reliable.", style="warning")
    if gof:
        rows, zero, zp = _gof_rows(obs, exp_, J - 1 + k + m)
        section.table(Table("Goodness-of-Fit", ["", "Chi-Square", "df", "Sig."], rows,
                            footnotes=[f"Link function: {LINKS[link]}."]))
        if zero:
            section.note("Warning", f"There are {zero} ({fmt(zp, 1)}%) cells (i.e., dependent "
                         "variable levels by observed combinations of predictor variable "
                         "values) with zero frequencies.", style="warning")
    if summary:
        pseudo = _pseudo_table(ll0, ll1, n)
        pseudo.footnotes = [f"Link function: {LINKS[link]}."]
        section.table(pseudo)

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = stats.norm.ppf(0.5 + level / 2)
    ci_g = f"{pct(level)} Confidence Interval"
    pnames = [f"[{dependent} = {format_code(data.levels[i])}]" for i in range(J - 1)]
    from rsstatchem.core.stats import boot_models as bm
    info = bm.ordinal_model(data.y, X, J, link,
                            [("Threshold" if i == 0 else "", pnames[i]) for i in range(J - 1)]
                            + [("Location" if i == 0 else "", c)
                               for i, c in enumerate(data.X.columns)], th, beta,
                            se=se[:J - 1 + k], rows=data.frame.index.to_numpy(),
                            table="Parameter Estimates", headers=["", ""],
                            est_label="Estimate", stat="wald", stat_label="Wald")
    if m:
        info.bootable = False
        info.note = "Bootstrap results are not available for models with a scale component."
    result.models = [info]
    if estimates:
        rows = []
        for i in range(J - 1):
            rows.append(_plum_row("Threshold" if i == 0 else "", pnames[i], th[i], se[i], z))
        all_cols = list(data.X.columns)
        first = True
        for name, cols, all_names in data.terms:
            for nm in all_names:
                head = "Location" if first else ""
                first = False
                if nm in cols:
                    i = J - 1 + all_cols.index(nm)
                    rows.append(_plum_row(head, nm, beta[i - (J - 1)], se[i], z))
                else:
                    rows.append([head, nm, "0" + sup("a"), "", "", "0", "", "", ""])
        zc = list(Zdf.columns)
        first = True
        for name, cols, all_names in zterms:
            for nm in all_names:
                head = "Scale" if first else ""
                first = False
                if nm in cols:
                    i = J - 1 + k + zc.index(nm)
                    rows.append(_plum_row(head, nm, tau[i - (J - 1 + k)], se[i], z))
                else:
                    rows.append([head, nm, "0" + sup("a"), "", "", "0", "", "", ""])
        redundant = any(len(c) < len(a) for _, c, a in data.terms + zterms)
        section.table(Table("Parameter Estimates", ["", "", "Estimate", "Std. Error", "Wald",
                                                    "df", "Sig.", (ci_g, "Lower Bound"),
                                                    (ci_g, "Upper Bound")], rows,
                            row_header_cols=2, footnotes=[f"Link function: {LINKS[link]}."]
                            + (["a. This parameter is set to zero because it is redundant."]
                               if redundant else [])))
    if correlations or covariances:
        names = pnames + list(data.X.columns) + [f"Scale: {c}" for c in Zdf.columns]
        if covariances:
            section.table(_matrix_table("Asymptotic Covariance Matrix", names, cov))
        if correlations:
            sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
            section.table(_matrix_table("Asymptotic Correlation Matrix", names,
                                        cov / np.outer(sd, sd)))
    if cell_info:
        section.table(_cell_table(ds, data, prob, pat, factors + covariates, "Cell Information"))
    if parallel and k and J > 2 and not m:
        llg = _general_ordinal_ll(data.y, X, J, link, th, beta)
        llg = max(llg, ll1)
        c = 2 * (llg - ll1)
        d = k * (J - 2)
        section.table(Table("Test of Parallel Linesᵃ", ["Model", "-2 Log Likelihood",
                                                        "Chi-Square", "df", "Sig."], [
            ["Null Hypothesis", fmt(-2 * (ll1 + const)), "", "", ""],
            ["General", fmt(-2 * (llg + const)) + sup("b"), fmt(c), fmt_int(d),
             _sig(stats.chi2.sf(c, d))]], footnotes=[
            "The null hypothesis states that the location parameters (slope coefficients) are "
            "the same across response categories.", f"a. Link function: {LINKS[link]}.",
            "b. General model: separate location parameters for each cumulative split."]))
    if classification:
        section.table(_class_table(data, prob))
    if save:
        all_terms = loc_terms + [t for t in scale_terms if t not in loc_terms]
        loc_cols = list(data.X.columns)
        zc = list(Zdf.columns)
        cols = list(dict.fromkeys(loc_cols + zc))

        def predict(Xn):
            frame = pd.DataFrame(Xn, columns=cols)
            return _ordinal_probs(th, beta, frame[loc_cols].to_numpy(float), link, tau,
                                  frame[zc].to_numpy(float) if zc else None)
        _save_category_probs(ds, data, prob, save, result, predict, all_terms, cols,
                             dependent)
    return result


def _plum_row(head, name, b, se, z) -> list[str]:
    w = (b / se) ** 2 if se > 0 else float("nan")
    return [head, name, fmt(b), fmt(se), fmt(w), "1", _sig(float(stats.chi2.sf(w, 1))),
            fmt(b - z * se), fmt(b + z * se)]

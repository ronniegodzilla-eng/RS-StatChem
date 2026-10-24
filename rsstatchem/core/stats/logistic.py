"""Analyze > Regression > Binary Logistic, Multinomial Logistic and Ordinal (SPSS LOGISTIC
REGRESSION, NOMREG and PLUM).

Binary logistic regression uses its own Newton-Raphson fit so the SPSS-specific pieces (Block 0
score tests, Forward/Backward LR and Wald selection, Hosmer-Lemeshow deciles of risk) can be
reproduced; estimates agree with ``statsmodels.Logit``. Multinomial uses ``statsmodels.MNLogit``
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

METHODS = {"enter": "Enter", "fstep_lr": "Forward: LR", "fstep_wald": "Forward: Wald",
           "bstep_lr": "Backward: LR", "bstep_wald": "Backward: Wald"}
CONTRASTS = ("indicator", "deviation")
LINKS = {"logit": "Logit", "probit": "Probit", "cloglog": "Complementary Log-log",
         "nloglog": "Negative Log-log", "cauchit": "Cauchit"}


@dataclass
class LogisticResult:
    section: Section
    saved: dict[str, tuple[str, pd.Series]] = field(default_factory=dict)


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


def _coding(ds: Dataset, name: str, values: pd.Series, contrast: str,
            reference: str) -> _Coding:
    var = ds.var(name)
    cats = sorted(values.dropna().unique(), key=_sort_key)
    k = len(cats)
    if contrast == "deviation":
        m = np.zeros((k, k - 1))
        rows = list(range(k - 1)) if reference == "last" else list(range(1, k))
        for j, r in enumerate(rows):
            m[r, j] = 1
        m[k - 1 if reference == "last" else 0, :] = -1
    else:
        m = np.zeros((k, k - 1))
        rows = list(range(k - 1)) if reference == "last" else list(range(1, k))
        for j, r in enumerate(rows):
            m[r, j] = 1
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


def binary_logistic(ds: Dataset, dependent: str, covariates: list[str],
                    categorical: dict[str, tuple[str, str]] | None = None,
                    method: str = "enter", cut: float = 0.5, p_enter: float = 0.05,
                    p_remove: float = 0.10, max_iter: int = 20, constant: bool = True,
                    hosmer: bool = True, ci: bool = True, level: float = 0.95,
                    iteration_history: bool = False, each_step: bool = True,
                    save=()) -> LogisticResult:
    """``categorical`` maps a covariate to (contrast, reference): ("indicator"|"deviation",
    "last"|"first"). ``save`` may contain "prob" and "group"."""
    categorical = dict(categorical or {})
    for name in covariates:
        if ds.var(name).type != NUMERIC and name not in categorical:
            categorical[name] = ("indicator", "last")
    dvar = ds.var(dependent)
    section = Section("Logistic Regression")
    cat_sub = "".join(f"\n  /CONTRAST ({c})={categorical[c][0].capitalize()}"
                      + ("(1)" if categorical[c][1] == "first" else "")
                      for c in covariates if c in categorical)
    meth_kw = {"enter": "ENTER", "fstep_lr": "FSTEP(LR)", "fstep_wald": "FSTEP(WALD)",
               "bstep_lr": "BSTEP(LR)", "bstep_wald": "BSTEP(WALD)"}[method]
    section.log(f"LOGISTIC REGRESSION VARIABLES {dependent}\n  /METHOD={meth_kw} "
                + " ".join(covariates) + cat_sub
                + ("\n  /SAVE=" + " ".join({"prob": "PRED", "group": "PGROUP"}[s]
                                          for s in save) if save else "")
                + "\n  /PRINT=" + " ".join(["GOODFIT"] * hosmer + ([f"CI({level * 100:g})"]
                                                                 if ci else [])
                                          + ["ITER(1)"] * iteration_history)
                + f"\n  /CRITERIA=PIN({p_enter:g}) POUT({p_remove:g}) ITERATE({max_iter})"
                  f" CUT({cut:g}).")
    result = LogisticResult(section)

    raw = {c: _raw(ds, c, c in categorical) for c in covariates}
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
    blocks, effects = [], []
    for c in covariates:
        if c in categorical:
            contrast, ref = categorical[c]
            coding = _coding(ds, c, frame[c], contrast, ref)
            if len(coding.categories) < 2:
                section.note("Warning", f"The categorical variable {c} has only one category "
                             "and was removed from the analysis.", style="warning")
                continue
            codings[c] = coding
            enc = _encode(frame[c], coding)
            blocks.append(enc)
            effects.append(_Effect(c, list(enc.columns), True))
        else:
            blocks.append(frame[[c]].astype(float))
            effects.append(_Effect(c, [c], False))
    if not effects:
        section.note("Warning", "No valid covariates.", style="warning")
        return result
    design = pd.concat(blocks, axis=1)
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
    ll0 = fit0.ll
    if not constant:
        ll0 = len(y) * math.log(0.5)
    p0 = probs(fit0)

    letters = iter("abcdefghijklmnopqrstuvwxyz")

    def classification(step_rows: list[tuple[str, np.ndarray]], title_cut: float) -> Table:
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
    section.table(Table("Variables not in the Equation", nc, not_in_rows(fit0, effects, "Step 0"),
                        row_header_cols=3))

    # ---------------------------------------------------------------- selection
    def removal_stat(step_effects: list[_Effect], f: _Fit, e: _Effect, lr: bool):
        if lr:
            reduced = fit([x for x in step_effects if x is not e])
            change = reduced.m2ll - f.m2ll
            return change, len(e.cols), float(stats.chi2.sf(change, len(e.cols))), reduced.ll
        idx = [f.cols.index(c) for c in e.cols]
        w = _wald(f.beta[idx], f.cov[np.ix_(idx, idx)])
        return w, len(e.cols), float(stats.chi2.sf(w, len(e.cols))), None

    steps: list[_Step] = []
    lr = method.endswith("_lr")
    if method == "enter" or method.startswith("bstep"):
        f1 = fit(effects)
        steps.append(_Step(list(effects), f1, [e.name for e in effects], []))
    if method != "enter":
        current = list(effects) if method.startswith("bstep") else []
        f = steps[-1].fit if steps else fit0
        last_change: tuple[str, str] | None = None
        for _ in range(4 * len(effects) + 4):
            changed = False
            order = ("remove", "enter") if method.startswith("bstep") else ("enter", "remove")
            for action in order:
                if action == "enter":
                    outside = [e for e in effects if e not in current]
                    best = None
                    for e in outside:
                        if last_change == ("remove", e.name):
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
                    if not current:
                        continue
                    worst = None
                    for e in current:
                        if last_change == ("enter", e.name) and len(current) == 1:
                            continue
                        _, _, p, _ = removal_stat(current, f, e, lr)
                        if worst is None or p > worst[1]:
                            worst = (e, p)
                    if worst is not None and worst[1] > p_remove and \
                            last_change != ("enter", worst[0].name):
                        current = [e for e in current if e is not worst[0]]
                        f = fit(current)
                        steps.append(_Step(list(current), f, [], [worst[0].name]))
                        last_change = ("remove", worst[0].name)
                        changed = True
                        break
            if not changed:
                break

    method_text = {"enter": "Enter", "fstep_lr": "Forward Stepwise (Likelihood Ratio)",
                   "fstep_wald": "Forward Stepwise (Wald)",
                   "bstep_lr": "Backward Stepwise (Likelihood Ratio)",
                   "bstep_wald": "Backward Stepwise (Wald)"}[method]
    section.note(f"Block 1: Method = {method_text}", "")
    if not steps:
        section.note("", "No variables were entered into the equation (no score statistic "
                     f"was significant at the {fmt(p_enter)} level).", style="warning")
        return result
    shown = steps if each_step else steps[-1:]
    first_no = len(steps) - len(shown) + 1

    if iteration_history:
        hrows = []
        for k, s in enumerate(shown, start=first_no):
            for i, (m2, b) in enumerate(s.fit.history):
                coefs = np.r_[b[-1:], b[:-1]] if constant else b
                hrows.append([f"Step {k}" if i == 0 else "", str(i + 1), fmt(m2, 3)]
                             + [fmt(v, 3) for v in coefs])
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
    prev_ll = fit0.ll if constant else ll0
    for k, s in enumerate(steps, start=1):
        f = s.fit
        df_model = len(f.cols) - (1 if constant else 0)
        prev_df = (len(steps[k - 2].fit.cols) - (1 if constant else 0)) if k > 1 else 0
        model_chi = 2 * (f.ll - ll0)
        step_chi = 2 * (f.ll - prev_ll)
        step_df = df_model - prev_df
        if k >= first_no:
            orows += [[f"Step {k}", "Step", fmt(step_chi), fmt_int(abs(step_df)),
                       _sig(float(stats.chi2.sf(abs(step_chi), abs(step_df))) if step_df
                            else float("nan"))],
                      ["", "Block", fmt(model_chi), fmt_int(df_model),
                       _sig(float(stats.chi2.sf(model_chi, df_model)))],
                      ["", "Model", fmt(model_chi), fmt_int(df_model),
                       _sig(float(stats.chi2.sf(model_chi, df_model)))]]
            cs, nk = _pseudo_r2(ll0, f.ll, len(y))
            srows.append([str(k), fmt(f.m2ll, 3) + sup("a"), fmt(cs), fmt(nk)])
            if hosmer:
                chi, hdf, hp, groups = hosmer_lemeshow(y, probs(f))
                hlrows.append([str(k), fmt(chi), fmt_int(hdf), _sig(hp)])
                for g, (m, o1, e1) in enumerate(groups, start=1):
                    hlcont.append([f"Step {k}" if g == 1 else "", str(g), fmt_int(m - o1),
                                   fmt(m - e1, 3), fmt_int(o1), fmt(e1, 3), fmt_int(m)])
        prev_ll = f.ll
    section.table(Table("Omnibus Tests of Model Coefficients", ["", "", "Chi-square", "df",
                                                                "Sig."], orows,
                        row_header_cols=2,
                        footnotes=["A negative Chi-squares value indicates that the "
                                   "Chi-squares value has decreased from the previous step."]
                        if any(r[2].startswith("-") for r in orows) else []))
    last = steps[-1].fit
    term = (f"a. Estimation terminated at iteration number {last.iterations} because "
            "parameter estimates changed by less than .001." if last.converged else
            f"a. Estimation terminated at iteration number {last.iterations} because maximum "
            "iterations has been reached. Final solution cannot be found.")
    section.table(Table("Model Summary", ["Step", "-2 Log likelihood", "Cox & Snell R Square",
                                          "Nagelkerke R Square"], srows, footnotes=[term]))
    if not last.converged:
        section.note("Warning", "The maximum number of iterations was reached; the estimates "
                     "may be infinite (complete or quasi-complete separation).", style="warning")
    if hosmer:
        section.table(Table("Hosmer and Lemeshow Test", ["Step", "Chi-square", "df", "Sig."],
                            hlrows))
        g0, g1 = f"{dvar.title} = {dlabels[0]}", f"{dvar.title} = {dlabels[1]}"
        section.table(Table("Contingency Table for Hosmer and Lemeshow Test",
                            ["", "", (g0, "Observed"), (g0, "Expected"), (g1, "Observed"),
                             (g1, "Expected"), "Total"], hlcont, row_header_cols=2))
    section.table(Table("Classification Tableᵃ", ccols, classification(
        [(f"Step {k}", probs(s.fit)) for k, s in enumerate(steps, start=1)
         if k >= first_no], cut), row_header_cols=3,
        footnotes=[f"a. The cut value is {fmt(cut, 3)}"]))

    # Variables in the Equation
    vcols_full = vcols + ([(f"{pct(level)} C.I.for EXP(B)", "Lower"),
                           (f"{pct(level)} C.I.for EXP(B)", "Upper")] if ci else [])
    z = stats.norm.ppf(0.5 + level / 2)
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
                              fmt(w), fmt_int(len(idx)), fmt_p(stats.chi2.sf(w, len(idx))),
                              ""] + (["", ""] if ci else []))
                first = False
            for i in idx:
                vrows.append(_coef_row(f"Step {k}{sup(letter)}" if first else "", f.cols[i],
                                       f.beta[i], se[i], ci, z))
                first = False
        if constant:
            vrows.append(_coef_row("", "Constant", f.beta[-1], se[-1], ci, z)[:8]
                         + (["", ""] if ci else []))
    section.table(Table("Variables in the Equation", vcols_full, vrows, row_header_cols=2,
                        footnotes=notes))

    if lr and method != "enter":
        mrows = []
        for k, s in enumerate(steps, start=1):
            if k < first_no or not s.effects:
                continue
            for j, e in enumerate(s.effects):
                change, df, p, ll_red = removal_stat(s.effects, s.fit, e, True)
                mrows.append([f"Step {k}" if j == 0 else "", e.name, fmt(ll_red, 3),
                              fmt(change, 3), fmt_int(df), fmt_p(p)])
        section.table(Table("Model if Term Removed", ["", "Variable", "Model Log Likelihood",
                                                      "Change in -2 Log Likelihood", "df",
                                                      "Sig. of the Change"], mrows,
                            row_header_cols=2))
    if method != "enter":
        nrows = []
        for k, s in enumerate(steps, start=1):
            outside = [e for e in effects if e not in s.effects]
            if k >= first_no and outside:
                nrows += not_in_rows(s.fit, outside, f"Step {k}")
        if nrows:
            section.table(Table("Variables not in the Equation", nc, nrows, row_header_cols=3))

    # ---------------------------------------------------------------- save
    if save:
        allcols = [c for e in steps[-1].effects for c in e.cols]
        parts = []
        for e in steps[-1].effects:
            if e.categorical:
                parts.append(_encode(ds.analysis_series(e.name), codings[e.name]))
            else:
                parts.append(numeric(ds, e.name).rename(e.name).to_frame())
        full = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ds.df.index)
        if constant:
            full["Constant"] = 1.0
        Xa = full[allcols + (["Constant"] if constant else [])].to_numpy(float)
        prob = pd.Series(special.expit(Xa @ last.beta), index=ds.df.index)
        if "prob" in save:
            result.saved["PRE"] = ("Predicted probability", prob)
        if "group" in save:
            codes = [v if isinstance(v, (int, float)) else np.nan for v in levels]
            grp = pd.Series(np.where(prob > cut, codes[1], codes[0]), index=ds.df.index)
            grp[prob.isna()] = np.nan
            result.saved["PGR"] = ("Predicted group", grp.astype(float))
    return result


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
    X: pd.DataFrame               # covariates then factor dummies (last level dropped)
    terms: list[tuple[str, list[str], list[str]]]   # effect, columns, all-level labels
    factor_levels: dict[str, list]
    n_missing: int
    frame: pd.DataFrame


def _prepare(ds: Dataset, dependent: str, factors: list[str], covariates: list[str]) -> _Data:
    frame = pd.DataFrame({"__y": ds.analysis_series(dependent),
                          **{f: ds.analysis_series(f) for f in factors},
                          **{c: numeric(ds, c) for c in covariates}})
    n_total = len(frame)
    frame = frame.dropna()
    levels = sorted(frame["__y"].unique(), key=_sort_key)
    dvar = ds.var(dependent)
    code = {v: i for i, v in enumerate(levels)}
    y = frame["__y"].map(code).to_numpy(int)
    cols, terms, flev = [], [], {}
    for c in covariates:
        cols.append(frame[c].astype(float).rename(c))
        terms.append((c, [c], [c]))
    for f in factors:
        lv = sorted(frame[f].unique(), key=_sort_key)
        flev[f] = lv
        names = [f"[{f}={format_code(v)}]" for v in lv]
        for v, nm in zip(lv[:-1], names[:-1]):
            cols.append((frame[f] == v).astype(float).rename(nm))
        terms.append((f, names[:-1], names))
    X = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=frame.index)
    return _Data(y, levels, [value_text(dvar, v) for v in levels], X, terms, flev,
                 n_total - len(frame), frame)


def _patterns(X: pd.DataFrame, y: np.ndarray, prob: np.ndarray, J: int):
    """Observed and expected counts per covariate pattern (SPSS subpopulations)."""
    if X.shape[1] == 0:
        key = np.zeros(len(y), dtype=int)
    else:
        _, key = np.unique(np.round(X.to_numpy(float), 10), axis=0, return_inverse=True)
        key = key.ravel()
    S = key.max() + 1
    obs = np.zeros((S, J))
    exp_ = np.zeros((S, J))
    np.add.at(obs, (key, y), 1)
    np.add.at(exp_, key, prob)
    return obs, exp_


def _multinomial_constant(obs: np.ndarray) -> float:
    return float(np.sum(special.gammaln(obs.sum(axis=1) + 1))
                 - np.sum(special.gammaln(obs + 1)))


def _gof_rows(obs, exp_, n_params) -> tuple[list[list[str]], int, float]:
    S, J = obs.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = float(np.nansum(np.where(exp_ > 0, (obs - exp_) ** 2 / exp_, 0)))
        dev = float(2 * np.nansum(np.where(obs > 0, obs * np.log(obs / exp_), 0)))
    df = S * (J - 1) - n_params
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


# ====================================================================== multinomial (NOMREG)
def _mnlogit(y: np.ndarray, X: np.ndarray, J: int):
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.MNLogit(y, X)
        try:
            res = model.fit(method="newton", maxiter=200, disp=0)
        except np.linalg.LinAlgError:
            res = model.fit(method="bfgs", maxiter=2000, disp=0)
    return res


def multinomial(ds: Dataset, dependent: str, factors: list[str] = (),
                covariates: list[str] = (), reference: str = "last", level: float = 0.95,
                classification: bool = True, gof: bool = True, lr_tests: bool = True,
                save=()) -> LogisticResult:
    """``reference`` is "last", "first" or the code of a category (as text)."""
    factors, covariates = list(factors), list(covariates)
    dvar = ds.var(dependent)
    section = Section("Nominal Regression")
    ref_kw = {"last": "LAST", "first": "FIRST"}.get(reference, reference)
    section.log(f"NOMREG {dependent} (BASE={ref_kw} ORDER=ASCENDING)"
                + (" BY " + " ".join(factors) if factors else "")
                + (" WITH " + " ".join(covariates) if covariates else "")
                + "\n  /CRITERIA CIN(" + f"{level * 100:g}" + ") DELTA(0) MXITER(100) "
                  "MXSTEP(5) CHKSEP(20) LCONVERGE(0) PCONVERGE(0.000001) SINGULAR(0.00000001)"
                + "\n  /MODEL\n  /INTERCEPT=INCLUDE\n  /PRINT=" + " ".join(
                    ["CLASSTABLE"] * classification + ["FIT"] * gof
                    + ["PARAMETER SUMMARY LRT CPS STEP MFI"]) + ".")
    result = LogisticResult(section)
    data = _prepare(ds, dependent, factors, covariates)
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

    def design(cols: list[str], intercept: bool = True) -> np.ndarray:
        X = data.X[cols].to_numpy(float) if cols else np.zeros((n, 0))
        return np.column_stack([np.ones(n), X]) if intercept else X

    all_cols = list(data.X.columns)
    res = _mnlogit(yr, design(all_cols), J)
    prob_r = np.asarray(res.predict(design(all_cols)))
    prob = prob_r[:, [order.index(j) for j in range(J)]]
    obs, exp_ = _patterns(data.X, data.y, prob, J)
    const = _multinomial_constant(obs)
    ll1 = float(res.llf)
    ll0 = _null_ll(data.y, J)
    n_par = (1 + len(all_cols)) * (J - 1)
    chi = 2 * (ll1 - ll0)
    df = n_par - (J - 1)
    section.table(Table("Model Fitting Information", [
        "Model", ("Model Fitting Criteria", "-2 Log Likelihood"),
        ("Likelihood Ratio Tests", "Chi-Square"), ("Likelihood Ratio Tests", "df"),
        ("Likelihood Ratio Tests", "Sig.")], [
        ["Intercept Only", fmt(-2 * (ll0 + const)), "", "", ""],
        ["Final", fmt(-2 * (ll1 + const)), fmt(chi), fmt_int(df),
         _sig(stats.chi2.sf(chi, df)) if df > 0 else ""]]))
    if gof:
        rows, zero, zp = _gof_rows(obs, exp_, n_par)
        section.table(Table("Goodness-of-Fit", ["", "Chi-Square", "df", "Sig."], rows))
        if zero:
            section.note("Warning", f"There are {zero} ({fmt(zp, 1)}%) cells (i.e., dependent "
                         "variable levels by subpopulations) with zero frequencies.",
                         style="warning")
    section.table(_pseudo_table(ll0, ll1, n))

    if lr_tests:
        lrows = []
        if factors:
            lrows.append(["Intercept", fmt(-2 * (ll1 + const)) + sup("b"), ".000", "0", ""])
        else:
            r0 = _mnlogit(yr, design(all_cols, intercept=False), J) if all_cols else None
            ll_r = float(r0.llf) if r0 is not None else n * math.log(1 / J)
            c = 2 * (ll1 - ll_r)
            lrows.append(["Intercept", fmt(-2 * (ll_r + const)), fmt(c), fmt_int(J - 1),
                          _sig(stats.chi2.sf(c, J - 1))])
        for name, cols, _ in data.terms:
            keep = [c for c in all_cols if c not in cols]
            r = _mnlogit(yr, design(keep), J)
            c = 2 * (ll1 - float(r.llf))
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
                "effect does not increase the degrees of freedom."] if factors else [])))

    params = np.asarray(res.params)
    bse = np.asarray(res.bse)
    z = stats.norm.ppf(0.5 + level / 2)
    ci_g = f"{pct(level)} Confidence Interval for Exp(B)"
    prow = []
    for k, j in enumerate(order[1:]):
        b_col, s_col = params[:, k], bse[:, k]
        head = data.labels[j] + sup("a") if k == 0 else data.labels[j]
        prow.append(_mn_row(head, "Intercept", b_col[0], s_col[0], z, exp=False))
        for name, cols, all_names in data.terms:
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
        + (["b. This parameter is set to zero because it is redundant."] if factors else [])))

    if classification:
        section.table(_class_table(data, prob))
    if save:
        _save_category_probs(ds, data, prob, save, result, factors, covariates,
                             lambda X: np.asarray(res.predict(
                                 np.column_stack([np.ones(len(X)), X])))[
                                 :, [order.index(j) for j in range(J)]])
    return result


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


def _save_category_probs(ds, data, prob, save, result, factors, covariates, predict) -> None:
    parts = {c: numeric(ds, c) for c in covariates}
    cols = []
    for c in covariates:
        cols.append(parts[c].rename(c))
    for f in factors:
        s = ds.analysis_series(f)
        for v in data.factor_levels[f][:-1]:
            col = (s == v).astype(float)
            col[s.isna() | ~s.isin(data.factor_levels[f])] = np.nan
            cols.append(col.rename(f"[{f}={format_code(v)}]"))
    X = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=ds.df.index)
    ok = X.notna().all(axis=1).to_numpy()
    full = np.full((ds.n_cases, len(data.levels)), np.nan)
    if ok.any():
        full[ok] = predict(X.to_numpy(float)[ok])
    index = ds.df.index
    if "pcp" in save or "est" in save:
        result.saved["EST"] = ("Estimated probability of the first response category",
                               pd.Series(full[:, 0], index=index))
    if "pre" in save or "group" in save:
        codes = [v if isinstance(v, (int, float)) else np.nan for v in data.levels]
        cat = np.full(ds.n_cases, np.nan)
        cat[ok] = np.asarray(codes, float)[np.nanargmax(full[ok], axis=1)]
        result.saved["PRE"] = ("Predicted response category", pd.Series(cat, index=index))
    if "pcp" in save or "max" in save:
        result.saved["PCP"] = ("Predicted category probability",
                               pd.Series(np.nanmax(full, axis=1) if ok.any() else full[:, 0],
                                         index=index))


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
    conv = bool(res.mle_retvals.get("converged", True))
    return th, beta, cov, float(res.llf), conv


def _ordinal_probs(th, beta, X, link) -> np.ndarray:
    eta = X @ beta if len(beta) else np.zeros(X.shape[0])
    cdf = _distr(link).cdf(th[None, :] - eta[:, None])
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
            parallel: bool = False, classification: bool = False, save=()) -> LogisticResult:
    factors, covariates = list(factors), list(covariates)
    dvar = ds.var(dependent)
    section = Section("PLUM - Ordinal Regression")
    section.log(f"PLUM {dependent}" + (" BY " + " ".join(factors) if factors else "")
                + (" WITH " + " ".join(covariates) if covariates else "")
                + f"\n  /CRITERIA=CIN({level * 100:g}) DELTA(0) LCONVERGE(0) MXITER(100) "
                  "MXSTEP(5) PCONVERGE(1.0E-6) SINGULAR(1.0E-8)"
                + f"\n  /LINK={link.upper()}\n  /PRINT=" + " ".join(
                    ["FIT"] * gof + ["PARAMETER SUMMARY"] + ["TPARALLEL"] * parallel) + ".")
    result = LogisticResult(section)
    data = _prepare(ds, dependent, factors, covariates)
    J = len(data.levels)
    n = len(data.y)
    if J < 2:
        section.note("Warning", "The dependent variable must have at least two observed "
                     "categories.", style="warning")
        return result
    section.table(_case_summary(ds, dependent, data, factors))
    X = data.X.to_numpy(float)
    k = X.shape[1]
    th, beta, cov, ll1, conv = _ordinal_fit(data.y, X, J, link)
    ll0 = _null_ll(data.y, J)
    prob = _ordinal_probs(th, beta, X, link)
    obs, exp_ = _patterns(data.X, data.y, prob, J)
    const = _multinomial_constant(obs)
    chi = 2 * (ll1 - ll0)
    section.table(Table("Model Fitting Information", [
        "Model", "-2 Log Likelihood", "Chi-Square", "df", "Sig."], [
        ["Intercept Only", fmt(-2 * (ll0 + const)), "", "", ""],
        ["Final", fmt(-2 * (ll1 + const)), fmt(chi), fmt_int(k),
         _sig(stats.chi2.sf(chi, k)) if k else ""]],
        footnotes=[f"Link function: {LINKS[link]}."]))
    if not conv:
        section.note("Warning", "The estimation did not converge; results may not be "
                     "reliable.", style="warning")
    if gof:
        rows, zero, zp = _gof_rows(obs, exp_, J - 1 + k)
        section.table(Table("Goodness-of-Fit", ["", "Chi-Square", "df", "Sig."], rows,
                            footnotes=[f"Link function: {LINKS[link]}."]))
        if zero:
            section.note("Warning", f"There are {zero} ({fmt(zp, 1)}%) cells (i.e., dependent "
                         "variable levels by observed combinations of predictor variable "
                         "values) with zero frequencies.", style="warning")
    pseudo = _pseudo_table(ll0, ll1, n)
    pseudo.footnotes = [f"Link function: {LINKS[link]}."]
    section.table(pseudo)

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = stats.norm.ppf(0.5 + level / 2)
    ci_g = f"{pct(level)} Confidence Interval"
    rows = []
    for i in range(J - 1):
        rows.append(_plum_row("Threshold" if i == 0 else "",
                              f"[{dependent} = {format_code(data.levels[i])}]", th[i], se[i], z))
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
    section.table(Table("Parameter Estimates", ["", "", "Estimate", "Std. Error", "Wald", "df",
                                                "Sig.", (ci_g, "Lower Bound"),
                                                (ci_g, "Upper Bound")], rows, row_header_cols=2,
                        footnotes=[f"Link function: {LINKS[link]}."]
                        + (["a. This parameter is set to zero because it is redundant."]
                           if factors else [])))
    if parallel and k and J > 2:
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
        tab = _class_table(data, prob)
        section.table(tab)
    if save:
        _save_category_probs(ds, data, prob, save, result, factors, covariates,
                             lambda Xn: _ordinal_probs(th, beta, Xn, link))
    return result


def _plum_row(head, name, b, se, z) -> list[str]:
    w = (b / se) ** 2 if se > 0 else float("nan")
    return [head, name, fmt(b), fmt(se), fmt(w), "1", _sig(float(stats.chi2.sf(w, 1))),
            fmt(b - z * se), fmt(b + z * se)]

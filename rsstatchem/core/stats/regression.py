"""Analyze > Regression > Linear (SPSS REGRESSION).

Mirrors the SPSS dialog: blocks of independents (Previous/Next) each with its own method
(Enter / Stepwise / Forward / Backward / Remove), selection variable with rule, case labels and
WLS weight. Statistics: estimates, confidence intervals, covariance matrix, model fit, R squared
change, descriptives, part and partial correlations, collinearity diagnostics (tolerance/VIF,
eigenvalues, condition indices, variance proportions), Durbin-Watson, casewise diagnostics and
PRESS. Plots: any pair of DEPENDNT / *ZPRED / *ZRESID / *DRESID / *ADJPRED / *SRESID / *SDRESID,
histogram, normal P-P and partial regression plots. Save: predicted values, residuals,
distances, prediction intervals and influence statistics with SPSS's formulas and names.

Every model is fitted from weighted means and cross-products, so listwise, pairwise and
mean-substitution missing-value handling and WLS share one code path (as SPSS REGRESSION does).
The classical assumption tests used in Indonesian theses (normality of residuals,
multicollinearity, Glejser / Breusch-Pagan / White, Durbin-Watson / Breusch-Godfrey / Runs) are
kept as an extra section.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import charts
from rsstatchem.core.stats.common import (fmt, fmt_int, fmt_p, numeric, pct, shapiro, sup,
                                          value_text)
from rsstatchem.core.stats.nonparametric import ks_details, ks_table, runs_details, runs_table

METHODS = ("enter", "stepwise", "forward", "backward", "remove")
METHOD_NAMES = {"enter": "Enter", "stepwise": "Stepwise", "forward": "Forward",
                "backward": "Backward", "remove": "Remove"}
DEFAULT_STATS = ("estimates", "model_fit", "durbin_watson", "collinearity")
ASSUMPTIONS = ("normality", "multicollinearity", "heteroscedasticity", "autocorrelation")
PLOT_VARS = {"DEPENDNT": "", "*ZPRED": "Regression Standardized Predicted Value",
             "*ZRESID": "Regression Standardized Residual",
             "*DRESID": "Regression Deleted (Press) Residual",
             "*ADJPRED": "Regression Adjusted (Press) Predicted Value",
             "*SRESID": "Regression Studentized Residual",
             "*SDRESID": "Regression Studentized Deleted (Press) Residual"}
# key -> (SPSS root name, label, SPSS /SAVE keyword)
SAVE_SPECS = {
    "pred": ("PRE", "Unstandardized Predicted Value", "PRED"),
    "zpred": ("ZPR", "Standardized Predicted Value", "ZPRED"),
    "adjpred": ("ADJ", "Adjusted Predicted Value", "ADJPRED"),
    "sepred": ("SEP", "Standard Error of Predicted Value", "SEPRED"),
    "resid": ("RES", "Unstandardized Residual", "RESID"),
    "zresid": ("ZRE", "Standardized Residual", "ZRESID"),
    "sresid": ("SRE", "Studentized Residual", "SRESID"),
    "dresid": ("DRE", "Deleted Residual", "DRESID"),
    "sdresid": ("SDR", "Studentized Deleted Residual", "SDRESID"),
    "mahal": ("MAH", "Mahalanobis Distance", "MAHAL"),
    "cook": ("COO", "Cook's Distance", "COOK"),
    "lever": ("LEV", "Centered Leverage Value", "LEVER"),
    "dfbeta": ("DFB", "DFBETA", "DFBETA"),
    "sdfbeta": ("SDB", "Standardized DFBETA", "SDBETA"),
    "dffit": ("DFF", "DFFIT", "DFFIT"),
    "sdffit": ("SDF", "Standardized DFFIT", "SDFIT"),
    "covratio": ("COV", "COVRATIO", "COVRATIO"),
    "mcin": ("MCI", "Mean prediction interval", "MCIN"),
    "icin": ("ICI", "Individual prediction interval", "ICIN"),
}
RELATIONS = {"eq": ("=", "Equal to"), "ne": ("~=", "Not equal to"), "lt": ("<", "Less than"),
             "le": ("<=", "Less than or equal to"), "gt": (">", "Greater than"),
             "ge": (">=", "Greater than or equal to")}
LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class RegressionResult:
    section: Section
    saved: dict[str, tuple] = field(default_factory=dict)


# ====================================================================== moments and fits
@dataclass
class Moments:
    names: list[str]
    mean: np.ndarray
    cp: np.ndarray          # centred (weighted) cross-products
    n: int                  # number of cases (degrees-of-freedom basis)
    wsum: float             # sum of weights (= n without WLS)

    def ix(self, cols) -> list[int]:
        return [self.names.index(c) for c in cols]


def moments(frame: pd.DataFrame, weights: np.ndarray | None = None) -> Moments:
    X = frame.to_numpy(float)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    wsum = float(w.sum())
    mean = (w[:, None] * X).sum(axis=0) / wsum if len(X) else np.zeros(X.shape[1])
    D = X - mean
    return Moments(list(frame.columns), mean, (D * w[:, None]).T @ D, len(X), wsum)


def pairwise_moments(frame: pd.DataFrame) -> Moments:
    """SPSS /MISSING PAIRWISE: pairwise covariances, all-valid means, minimum pairwise N."""
    valid = frame.notna().astype(float)
    counts = valid.T @ valid
    n = int(counts.to_numpy().min())
    cov = frame.cov(min_periods=2).to_numpy(float)
    return Moments(list(frame.columns), frame.mean().to_numpy(float), cov * (n - 1), n,
                   float(n))


@dataclass
class Fit:
    cols: list[str]
    constant: bool
    b: np.ndarray            # [const] + slopes
    cov: np.ndarray
    df_model: int
    df_resid: int
    ess: float
    ssr: float
    n: int

    @property
    def names(self) -> list[str]:
        return (["const"] if self.constant else []) + list(self.cols)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))

    @property
    def t(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.b / self.se

    @property
    def p(self) -> np.ndarray:
        return 2 * stats.t.sf(np.abs(self.t), max(self.df_resid, 1))

    @property
    def tss(self) -> float:
        return self.ess + self.ssr

    @property
    def r2(self) -> float:
        return self.ess / self.tss if self.tss > 0 else float("nan")

    @property
    def r2_adj(self) -> float:
        n_eff = self.df_resid + self.df_model + (1 if self.constant else 0)
        base = n_eff - 1 if self.constant else n_eff
        return 1 - (1 - self.r2) * base / self.df_resid if self.df_resid > 0 else float("nan")

    @property
    def mse(self) -> float:
        return self.ssr / self.df_resid if self.df_resid > 0 else float("nan")

    @property
    def fvalue(self) -> float:
        if not self.df_model or not self.df_resid or self.mse <= 0:
            return float("nan")
        return (self.ess / self.df_model) / self.mse

    @property
    def f_pvalue(self) -> float:
        f = self.fvalue
        return float(stats.f.sf(f, self.df_model, self.df_resid)) if f == f else float("nan")

    def coef(self, name: str) -> int:
        return self.names.index(name)


def _solve_inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def fit_moments(M: Moments, dep: str, cols: list[str], constant: bool = True) -> Fit:
    iy = M.names.index(dep)
    ix = M.ix(cols)
    k = len(cols)
    if constant:
        S = M.cp
        syy = float(S[iy, iy])
        if k:
            Sxx = S[np.ix_(ix, ix)]
            sxy = S[ix, iy]
            inv = _solve_inv(Sxx)
            slopes = inv @ sxy
            ess = float(slopes @ sxy)
        else:
            inv, slopes, ess = np.zeros((0, 0)), np.zeros(0), 0.0
        df_resid = M.n - k - 1
        ssr = max(syy - ess, 0.0)
        mse = ssr / df_resid if df_resid > 0 else float("nan")
        xbar = M.mean[ix]
        b0 = float(M.mean[iy] - slopes @ xbar)
        cov_s = mse * inv
        cov = np.zeros((k + 1, k + 1))
        cov[1:, 1:] = cov_s
        cov[0, 0] = mse / M.wsum + float(xbar @ cov_s @ xbar)
        cov[0, 1:] = cov[1:, 0] = -cov_s @ xbar
        return Fit(list(cols), True, np.r_[b0, slopes], cov, k, df_resid, ess, ssr, M.n)
    U = M.cp + M.wsum * np.outer(M.mean, M.mean)
    uyy = float(U[iy, iy])
    if k:
        inv = _solve_inv(U[np.ix_(ix, ix)])
        b = inv @ U[ix, iy]
        ess = float(b @ U[ix, iy])
    else:
        inv, b, ess = np.zeros((0, 0)), np.zeros(0), 0.0
    df_resid = M.n - k
    ssr = max(uyy - ess, 0.0)
    mse = ssr / df_resid if df_resid > 0 else float("nan")
    return Fit(list(cols), False, b, mse * inv, k, df_resid, ess, ssr, M.n)


def tolerances(M: Moments, cols: list[str]) -> np.ndarray:
    """1 - R^2 of each predictor on the others (from the correlation matrix)."""
    if not cols:
        return np.zeros(0)
    S = M.cp[np.ix_(M.ix(cols), M.ix(cols))]
    d = np.sqrt(np.diag(S))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / np.outer(d, d)
    return 1 / np.diag(_solve_inv(R))


def std_beta(M: Moments, dep: str, fit: Fit) -> dict[str, float]:
    syy = M.cp[M.names.index(dep)][M.names.index(dep)]
    out = {}
    for c in fit.cols:
        j = M.names.index(c)
        out[c] = float(fit.b[fit.coef(c)] * math.sqrt(M.cp[j, j] / syy)) if syy > 0 \
            else float("nan")
    return out


def collinearity_diagnostics(M: Moments, cols: list[str], constant: bool = True):
    """Eigenvalues of the unit-length-scaled (uncentred) cross-product matrix, condition
    indices and variance-decomposition proportions (Belsley, Kuh and Welsch; as SPSS)."""
    ix = M.ix(cols)
    U = M.cp[np.ix_(ix, ix)] + M.wsum * np.outer(M.mean[ix], M.mean[ix])
    if constant:
        top = np.r_[M.wsum, M.wsum * M.mean[ix]]
        A = np.zeros((len(ix) + 1, len(ix) + 1))
        A[0] = top
        A[:, 0] = top
        A[1:, 1:] = U
    else:
        A = U
    d = 1 / np.sqrt(np.diag(A))
    lam, V = np.linalg.eigh(A * np.outer(d, d))
    order = np.argsort(lam)[::-1]
    lam, V = np.clip(lam[order], 1e-300, None), V[:, order]
    cond = np.sqrt(lam[0] / lam)
    phi = V ** 2 / lam[None, :]
    props = phi / phi.sum(axis=1, keepdims=True)
    return lam, cond, props.T          # props[dimension, coefficient]


# ====================================================================== stepping
@dataclass
class Step:
    cols: list[str]
    entered: list[str]
    removed: list[str]
    method: str
    block: int
    fit: Fit | None = None


def _enter_stat(M, dep, cols, c, constant):
    f = fit_moments(M, dep, cols + [c], constant)
    t = f.t[f.coef(c)]
    F = float(t * t)
    return F, float(stats.f.sf(F, 1, max(f.df_resid, 1))), f


def _select_block(M, dep, current, block_vars, method, crit, constant, tol):
    """Steps of one block. ``crit`` = (use_f, p_enter, p_remove, f_enter, f_remove)."""
    use_f, p_in, p_out, f_in, f_out = crit
    steps: list[Step] = []
    if method == "enter":
        new = [c for c in block_vars if c not in current]
        ok = []
        for c in new:
            if tolerances(M, current + ok + [c])[-1] >= tol:
                ok.append(c)
        return [Step(current + ok, ok, [], method, 0)] if ok else []
    if method == "remove":
        gone = [c for c in block_vars if c in current]
        return [Step([c for c in current if c not in gone], [], gone, method, 0)] if gone \
            else []
    cur = list(current)
    if method == "backward":
        new = [c for c in block_vars if c not in cur]
        if new:
            cur = cur + new
            steps.append(Step(list(cur), new, [], "enter", 0))
    last: tuple[str, str] | None = None
    for _ in range(4 * len(block_vars) + 4):
        changed = False
        if method in ("forward", "stepwise"):
            best = None
            for c in block_vars:
                if c in cur or last == ("remove", c):
                    continue
                tl = tolerances(M, cur + [c])
                if tl.size and tl.min() < tol:
                    continue
                F, p, _ = _enter_stat(M, dep, cur, c, constant)
                key = -F if use_f else p
                if best is None or key < best[0]:
                    best = (key, c, F, p)
            if best is not None and (best[2] >= f_in if use_f else best[3] <= p_in):
                cur = cur + [best[1]]
                steps.append(Step(list(cur), [best[1]], [], method, 0))
                last = ("enter", best[1])
                changed = True
        if method in ("backward", "stepwise"):
            f = fit_moments(M, dep, cur, constant)
            cands = [c for c in cur if c in block_vars and last != ("enter", c)]
            worst = None
            for c in cands:
                t = f.t[f.coef(c)]
                F = float(t * t)
                p = float(stats.f.sf(F, 1, max(f.df_resid, 1)))
                key = F if use_f else -p
                if worst is None or key < worst[0]:
                    worst = (key, c, F, p)
            if worst is not None and (worst[2] <= f_out if use_f else worst[3] >= p_out):
                cur = [c for c in cur if c != worst[1]]
                steps.append(Step(list(cur), [], [worst[1]], method, 0))
                last = ("remove", worst[1])
                changed = True
        if not changed:
            break
    return steps


# ====================================================================== case diagnostics
def case_diagnostics(fit: Fit, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                     level: float = 0.95) -> dict[str, np.ndarray]:
    """SPSS REGRESSION residual and influence statistics for the analysis cases.

    ``X`` holds the predictor columns of ``fit.cols`` (no constant column)."""
    Xd = np.column_stack([np.ones(len(y)), X]) if fit.constant else X
    p = Xd.shape[1]
    mse = fit.mse
    s = math.sqrt(mse) if mse > 0 else float("nan")
    G = fit.cov / mse if mse > 0 else np.zeros_like(fit.cov)
    pred = Xd @ fit.b
    e = y - pred
    htil = np.einsum("ij,jk,ik->i", Xd, G, Xd)
    h = w * htil
    one_h = np.clip(1 - h, 1e-12, None)
    wsum = w.sum()
    mp = np.sum(w * pred) / wsum
    sdp = math.sqrt(np.sum(w * (pred - mp) ** 2) / (wsum - 1)) if wsum > 1 else float("nan")
    df = fit.df_resid
    sw = np.sqrt(w)
    zresid = sw * e / s
    sresid = sw * e / (s * np.sqrt(one_h))
    dresid = e / one_h
    with np.errstate(invalid="ignore", divide="ignore"):
        sdresid = sresid * np.sqrt((df - 1) / np.clip(df - sresid ** 2, 1e-12, None))
        s_i = np.sqrt(np.clip((fit.ssr - w * e ** 2 / one_h) / (df - 1), 0, None))
    if fit.constant:
        mahal = (wsum - 1) * (htil - 1 / wsum)
        lever = h - w / wsum
    else:
        mahal = (wsum - 1) * htil
        lever = h
    dfbeta = (Xd @ G) * (w * e / one_h)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        sdfbeta = dfbeta / (s_i[:, None] * np.sqrt(np.diag(G))[None, :])
        covratio = (s_i ** 2 / mse) ** p / one_h
    tq = stats.t.ppf(0.5 + level / 2, max(df, 1))
    sepred = s * np.sqrt(htil)
    ind = np.sqrt(mse / w + sepred ** 2)
    return {"pred": pred, "resid": e, "zpred": (pred - mp) / sdp if sdp > 0 else pred * 0,
            "sepred": sepred, "zresid": zresid, "sresid": sresid, "dresid": dresid,
            "adjpred": y - dresid, "sdresid": sdresid, "mahal": mahal,
            "cook": sresid ** 2 * h / (one_h * p), "lever": lever, "dfbeta": dfbeta,
            "sdfbeta": sdfbeta, "dffit": h * e / one_h,
            "sdffit": sdresid * np.sqrt(h / one_h), "covratio": covratio,
            "lmci": pred - tq * sepred, "umci": pred + tq * sepred,
            "lici": pred - tq * ind, "uici": pred + tq * ind, "h": h}


def _select_mask(values: pd.Series, relation: str, value) -> pd.Series:
    if relation in ("eq", "ne"):
        hit = values == value
        if relation == "ne":
            hit = (values != value) & values.notna()
        return hit.fillna(False)
    num = pd.to_numeric(values, errors="coerce")
    v = float(value)
    op = {"lt": num < v, "le": num <= v, "gt": num > v, "ge": num >= v}[relation]
    return op.fillna(False)


# ====================================================================== main entry
def linear(ds: Dataset, dependent: str, independents: list[str] | None = None,
           method: str = "enter", statistics=DEFAULT_STATS,
           plots=("histogram", "pp", "scatter"), save=(), assumptions=ASSUMPTIONS,
           level: float = 0.95, p_enter: float = 0.05, p_remove: float = 0.10, *,
           blocks: list[tuple[str, list[str]]] | None = None,
           selection: tuple[str, str, object] | None = None, case_labels: str | None = None,
           wls: str | None = None, use_f: bool = False, f_enter: float = 3.84,
           f_remove: float = 2.71, constant: bool = True, missing: str = "listwise",
           casewise_sd: float = 3.0, casewise_all: bool = False,
           plot_pairs=(), partial_plots: bool = False,
           tolerance: float = 0.0001, pi_level: float | None = None) -> RegressionResult:
    """``blocks`` is a list of (method, variables); without it one block (method,
    independents) is used. ``selection`` = (variable, relation, value) with relation one of
    eq/ne/lt/le/gt/ge. ``missing`` is listwise / pairwise / mean. ``plot_pairs`` lists
    (Y, X) pairs of :data:`PLOT_VARS` keys; "scatter" in ``plots`` adds (*ZRESID, *ZPRED)."""
    from statsmodels.stats.stattools import durbin_watson

    if blocks is None:
        blocks = [(method, list(independents or []))]
    blocks = [(m, list(v)) for m, v in blocks if v]
    allvars = list(dict.fromkeys(v for _, vs in blocks for v in vs))
    statistics, plots, save = set(statistics), set(plots), set(save)
    pairs = list(dict.fromkeys(tuple(p) for p in plot_pairs))
    if "scatter" in plots and ("*ZRESID", "*ZPRED") not in pairs:
        pairs.append(("*ZRESID", "*ZPRED"))
    dvar = ds.var(dependent)
    titles = {n: ds.var(n).title for n in allvars}
    section = Section("Regression")
    result = RegressionResult(section)
    section.log(_syntax(dependent, blocks, statistics, plots, pairs, save, level, p_enter,
                        p_remove, use_f, f_enter, f_remove, constant, missing, selection,
                        wls, casewise_sd, casewise_all, partial_plots))
    if not allvars:
        section.note("Warning", "No independent variables were specified.", style="warning")
        return result

    # ---------------------------------------------------------------- data
    names = [dependent] + allvars
    raw = pd.DataFrame({n: numeric(ds, n) for n in dict.fromkeys(names)})
    wser = numeric(ds, wls) if wls else pd.Series(1.0, index=raw.index)
    selected = pd.Series(True, index=raw.index)
    sel_text = ""
    if selection:
        svar, rel, sval = selection
        selected = _select_mask(ds.analysis_series(svar), rel, sval)
        sel_text = f"{ds.var(svar).name} {RELATIONS[rel][0]} {format_code(sval)}"
    wok = wser.notna() & (wser > 0)
    if missing == "mean":
        base = raw[selected & wok & raw[dependent].notna()]
        filled = raw.fillna(base.mean())
        complete = filled.notna().all(axis=1) & raw[dependent].notna() & wok
        data = filled
    else:
        complete = raw.notna().all(axis=1) & wok
        data = raw
    use = complete & selected
    frame = data[use]
    w = wser[use].to_numpy(float)
    n = len(frame)
    if missing == "pairwise":
        M = pairwise_moments(raw[selected & wok])
    else:
        M = moments(frame, w if wls else None)
    if M.n < len(allvars) + (2 if constant else 1):
        section.note("Warning", "Not enough valid cases for the regression.", style="warning")
        return result

    # ---------------------------------------------------------------- stepping
    crit = (use_f, p_enter, p_remove, f_enter, f_remove)
    steps: list[Step] = []
    current: list[str] = []
    for bi, (meth, bvars) in enumerate(blocks, start=1):
        for s in _select_block(M, dependent, current, bvars, meth, crit, constant, tolerance):
            s.block = bi
            steps.append(s)
            current = s.cols
    if not steps:
        section.note("Warning", "No variables were entered into the equation.", style="warning")
        return result
    for s in steps:
        s.fit = fit_moments(M, dependent, s.cols, constant)

    dep_note = f"Dependent Variable: {dvar.title}"
    wls_note = f"Weighted Least Squares Regression - Weighted by {ds.var(wls).title}" \
        if wls else ""
    sel_note = f"Selecting only cases for which {sel_text}." if selection else ""

    # ---------------------------------------------------------------- descriptives
    if "descriptives" in statistics:
        rows = []
        for c in names:
            j = M.names.index(c)
            sd = math.sqrt(M.cp[j, j] / (M.wsum - 1)) if M.wsum > 1 else float("nan")
            nn = M.n if missing != "pairwise" else int(raw[c][selected & wok].notna().sum())
            rows.append([ds.var(c).title, fmt(M.mean[j], ds.var(c).decimals + 2),
                         fmt(sd, ds.var(c).decimals + 3), fmt_int(nn)])
        section.table(Table("Descriptive Statistics", ["", "Mean", "Std. Deviation", "N"], rows,
                            footnotes=[f"a. {wls_note}"] if wls else []))
        ix = M.ix(names)
        S = M.cp[np.ix_(ix, ix)]
        dd = np.sqrt(np.diag(S))
        R = S / np.outer(dd, dd)
        crow = []
        heads = [ds.var(c).title for c in names]
        for label in ("Pearson Correlation", "Sig. (1-tailed)", "N"):
            for i, c in enumerate(names):
                vals = []
                for j in range(len(names)):
                    r = R[i, j]
                    if label == "Pearson Correlation":
                        vals.append(fmt(r))
                    elif label == "N":
                        vals.append(fmt_int(M.n))
                    elif i == j:
                        vals.append(".")
                    else:
                        df = M.n - 2
                        tt = r * math.sqrt(df / max(1 - r * r, 1e-300))
                        vals.append(fmt_p(stats.t.sf(abs(tt), df)))
                crow.append([label if i == 0 else "", heads[i]] + vals)
        section.table(Table("Correlations", ["", ""] + heads, crow, row_header_cols=2))

    # ---------------------------------------------------------------- entered/removed
    crit_text = {
        "enter": "Enter", "remove": "Remove",
        "stepwise": (f"Stepwise (Criteria: F-to-enter >= {fmt(f_enter)}, F-to-remove <= "
                     f"{fmt(f_remove)})." if use_f else
                     f"Stepwise (Criteria: Probability-of-F-to-enter <= {fmt(p_enter)}, "
                     f"Probability-of-F-to-remove >= {fmt(p_remove)})."),
        "forward": (f"Forward (Criterion: F-to-enter >= {fmt(f_enter)})" if use_f else
                    f"Forward (Criterion: Probability-of-F-to-enter <= {fmt(p_enter)})"),
        "backward": (f"Backward (criterion: F-to-remove <= {fmt(f_remove)})." if use_f else
                     f"Backward (criterion: Probability of F-to-remove >= {fmt(p_remove)})."),
    }
    erows = [[str(i), ", ".join(titles[c] for c in s.entered) or ".",
              ", ".join(titles[c] for c in s.removed) or ".", crit_text[s.method]]
             for i, s in enumerate(steps, start=1)]
    enotes = [f"a. {dep_note}"]
    if any(s.method == "enter" for s in steps):
        enotes.append("b. All requested variables entered.")
    if wls:
        enotes.append(f"{wls_note}")
    section.table(Table("Variables Entered/Removedᵃ", ["Model", "Variables Entered",
                                                       "Variables Removed", "Method"],
                        erows, footnotes=enotes))

    def predictors(s: Step) -> str:
        parts = (["(Constant)"] if constant else []) + [titles[c] for c in reversed(s.cols)]
        return ", ".join(parts) if parts else "(none)"

    pred_notes = [f"{LETTERS[i % 26]}. Predictors: {predictors(s)}"
                  for i, s in enumerate(steps)]
    after = len(steps)
    extra_notes = []
    if not constant:
        extra_notes.append("For regression through the origin (the no-intercept model), R "
                           "Square measures the proportion of the variability in the "
                           "dependent variable about the origin explained by regression. This "
                           "CANNOT be compared to R Square for models which include an "
                           "intercept.")
    tail = [f"{LETTERS[(after + k) % 26]}. {t}" for k, t in
            enumerate([dep_note] + ([wls_note] if wls else []) + ([sel_note] if sel_note
                                                                  else []) + extra_notes)]

    # ---------------------------------------------------------------- diagnostics (final)
    final = steps[-1]
    ffit = final.fit
    Xa = frame[final.cols].to_numpy(float)
    ya = frame[dependent].to_numpy(float)
    pil = pi_level or level
    diag = case_diagnostics(ffit, Xa, ya, w, pil)

    # ---------------------------------------------------------------- model summary
    if "model_fit" in statistics:
        unsel = complete & ~selected
        cols = ["Model"]
        if selection:
            cols += [("R", f"{sel_text} (Selected)"),
                     ("R", f"{sel_text.replace(' = ', ' ~= ')} (Unselected)")]
        else:
            cols.append("R")
        cols += ["R Square", "Adjusted R Square", "Std. Error of the Estimate"]
        change = "r2_change" in statistics
        if change:
            cols += [("Change Statistics", h) for h in ("R Square Change", "F Change", "df1",
                                                        "df2", "Sig. F Change")]
        if "press" in statistics:
            cols.append("PRESS")
        if "durbin_watson" in statistics:
            cols.append("Durbin-Watson")
        rows, prev_r2, prev_k = [], 0.0, 0
        for i, s in enumerate(steps):
            f = s.fit
            r2 = f.r2
            row = [str(i + 1), fmt(math.sqrt(max(r2, 0))) + sup(LETTERS[i % 26])]
            if selection:
                ru = float("nan")
                if unsel.sum() > 2:
                    Xu = data.loc[unsel, s.cols].to_numpy(float)
                    pu = (np.column_stack([np.ones(len(Xu)), Xu]) if constant else Xu) @ f.b
                    yu = data.loc[unsel, dependent].to_numpy(float)
                    if np.ptp(pu) > 0 and np.ptp(yu) > 0:
                        ru = float(np.corrcoef(pu, yu)[0, 1])
                row.append(fmt(ru))
            row += [fmt(r2), fmt(f.r2_adj), fmt(math.sqrt(f.mse), dvar.decimals + 5)]
            if change:
                df1 = abs(len(s.cols) - prev_k) or 1
                d_r2 = r2 - prev_r2
                f_ch = (abs(d_r2) / df1) / ((1 - r2) / f.df_resid) if r2 < 1 and \
                    f.df_resid > 0 else float("nan")
                p_ch = stats.f.sf(f_ch, df1, f.df_resid) if f_ch == f_ch else float("nan")
                row += [fmt(d_r2), fmt(f_ch), fmt_int(df1), fmt_int(f.df_resid), fmt_p(p_ch)]
            if "press" in statistics:
                if i == len(steps) - 1:
                    row.append(fmt(float(np.sum(w * diag["dresid"] ** 2)), 3))
                else:
                    Xi = frame[s.cols].to_numpy(float)
                    di = case_diagnostics(f, Xi, ya, w, pil)
                    row.append(fmt(float(np.sum(w * di["dresid"] ** 2)), 3))
            if "durbin_watson" in statistics:
                row.append(fmt(durbin_watson(np.sqrt(w) * diag["resid"]))
                           if i == len(steps) - 1 else "")
            rows.append(row)
            prev_r2, prev_k = r2, len(s.cols)
        section.table(Table("Model Summary" + sup(LETTERS[after % 26]), cols, rows,
                            footnotes=pred_notes + tail))

        arows = []
        for i, s in enumerate(steps):
            f = s.fit
            dfm, dfr = f.df_model, f.df_resid
            arows += [[str(i + 1), "Regression", fmt(f.ess), fmt_int(dfm),
                       fmt(f.ess / dfm) if dfm else "", fmt(f.fvalue) if dfm else "",
                       (fmt_p(f.f_pvalue) + sup(LETTERS[(i + 1) % 26])) if dfm else ""],
                      ["", "Residual", fmt(f.ssr), fmt_int(dfr), fmt(f.mse), "", ""],
                      ["", "Total", fmt(f.tss), fmt_int(dfm + dfr), "", "", ""]]
        anotes = [f"a. {dep_note}"] + [f"{LETTERS[(i + 1) % 26]}. Predictors: {predictors(s)}"
                                       for i, s in enumerate(steps)]
        if wls:
            anotes.append(wls_note)
        if not constant:
            anotes.append("This total sum of squares is not corrected for the constant because "
                          "the constant is zero for regression through the origin.")
        section.table(Table("ANOVAᵃ", ["Model", "", "Sum of Squares", "df", "Mean Square", "F",
                                       "Sig."], arows, row_header_cols=2, footnotes=anotes))

    # ---------------------------------------------------------------- coefficients
    ci = "ci" in statistics
    pp = "part_partial" in statistics
    col = "collinearity" in statistics
    ccols = ["Model", "", ("Unstandardized Coefficients", "B"),
             ("Unstandardized Coefficients", "Std. Error"),
             ("Standardized Coefficients", "Beta"), "t", "Sig."]
    cig = f"{pct(level)} Confidence Interval for B"
    if ci:
        ccols += [(cig, "Lower Bound"), (cig, "Upper Bound")]
    if pp:
        ccols += [("Correlations", "Zero-order"), ("Correlations", "Partial"),
                  ("Correlations", "Part")]
    if col:
        ccols += [("Collinearity Statistics", "Tolerance"), ("Collinearity Statistics", "VIF")]
    iy = M.names.index(dependent)
    crows = []
    for i, s in enumerate(steps):
        f = s.fit
        betas = std_beta(M, dependent, f)
        tol = dict(zip(s.cols, tolerances(M, s.cols))) if col else {}
        first = True
        for name in f.names:
            j = f.coef(name)
            q = stats.t.ppf(0.5 + level / 2, max(f.df_resid, 1))
            row = [str(i + 1) if first else "",
                   "(Constant)" if name == "const" else titles[name], fmt(f.b[j]),
                   fmt(f.se[j]), "" if name == "const" else fmt(betas[name]), fmt(f.t[j]),
                   fmt_p(f.p[j])]
            first = False
            if ci:
                row += [fmt(f.b[j] - q * f.se[j]), fmt(f.b[j] + q * f.se[j])]
            if pp:
                if name == "const":
                    row += ["", "", ""]
                else:
                    jj = M.names.index(name)
                    zero = M.cp[jj, iy] / math.sqrt(M.cp[jj, jj] * M.cp[iy, iy])
                    tv = f.t[j]
                    partial = tv / math.sqrt(tv * tv + f.df_resid)
                    part = tv * math.sqrt(max(1 - f.r2, 0) / f.df_resid)
                    row += [fmt(zero), fmt(partial), fmt(part)]
            if col:
                tl = tol.get(name)
                row += [fmt(tl) if tl is not None else "",
                        fmt(1 / tl) if tl is not None and tl > 0 else ""]
            crows.append(row)
    section.table(Table("Coefficientsᵃ", ccols, crows, row_header_cols=2,
                        footnotes=[f"a. {dep_note}"] + ([f"b. {wls_note}"] if wls else [])))

    # ---------------------------------------------------------------- excluded variables
    xrows = []
    for i, s in enumerate(steps):
        out = [c for c in allvars if c not in s.cols]
        first = True
        for c in out:
            f2 = fit_moments(M, dependent, s.cols + [c], constant)
            j = f2.coef(c)
            tv = f2.t[j]
            tl = tolerances(M, s.cols + [c])
            row = [str(i + 1) if first else "", titles[c],
                   fmt(std_beta(M, dependent, f2)[c]) + sup(LETTERS[(i + 1) % 26]), fmt(tv),
                   fmt_p(f2.p[j]), fmt(tv / math.sqrt(tv * tv + f2.df_resid)), fmt(tl[-1])]
            if col:
                row += [fmt(1 / tl[-1]) if tl[-1] > 0 else "", fmt(tl.min())]
            xrows.append(row)
            first = False
    if xrows:
        xcols = ["Model", "", "Beta In", "t", "Sig.", "Partial Correlation",
                 ("Collinearity Statistics", "Tolerance")]
        if col:
            xcols += [("Collinearity Statistics", "VIF"),
                      ("Collinearity Statistics", "Minimum Tolerance")]
        section.table(Table("Excluded Variablesᵃ", xcols, xrows, row_header_cols=2,
                            footnotes=[f"a. {dep_note}"]
                            + [f"{LETTERS[(i + 1) % 26]}. Predictors in the Model: "
                               f"{predictors(s)}" for i, s in enumerate(steps)
                               if any(c not in s.cols for c in allvars)]))

    # ---------------------------------------------------------------- coefficient correlations
    if "bcov" in statistics:
        brows = []
        for i, s in enumerate(steps):
            f = s.fit
            order = list(reversed(s.cols))
            idx = [f.coef(c) for c in order]
            C = f.cov[np.ix_(idx, idx)]
            d = np.sqrt(np.diag(C))
            R = C / np.outer(d, d) if len(d) else C
            for label, Mx in (("Correlations", R), ("Covariances", C)):
                for a, ca in enumerate(order):
                    brows.append([str(i + 1) if (label == "Correlations" and a == 0) else "",
                                  label if a == 0 else "", titles[ca]]
                                 + [fmt(Mx[a, b]) for b in range(len(order))]
                                 + [""] * (max(len(x.cols) for x in steps) - len(order)))
        width = max(len(x.cols) for x in steps)
        heads = list(reversed(steps[-1].cols)) if steps[-1].cols else []
        heads = [titles[c] for c in heads] + [""] * (width - len(heads))
        section.table(Table("Coefficient Correlationsᵃ", ["Model", "", ""] + heads, brows,
                            row_header_cols=3, footnotes=[f"a. {dep_note}"]))

    # ---------------------------------------------------------------- collinearity diagnostics
    if col:
        drows = []
        width = max(len(s.cols) for s in steps) + (1 if constant else 0)
        for i, s in enumerate(steps):
            if not s.cols:
                continue
            lam, cond, props = collinearity_diagnostics(M, s.cols, constant)
            for d in range(len(lam)):
                drows.append([str(i + 1) if d == 0 else "", str(d + 1), fmt(lam[d]),
                              fmt(cond[d])] + [fmt(v, 2) for v in props[d]]
                             + [""] * (width - props.shape[1]))
        heads = (["(Constant)"] if constant else []) + \
            [titles[c] for c in max((s.cols for s in steps), key=len)]
        section.table(Table("Collinearity Diagnosticsᵃ", [
            "Model", "Dimension", "Eigenvalue", "Condition Index"]
            + [("Variance Proportions", h) for h in heads], drows, row_header_cols=2,
            footnotes=[f"a. {dep_note}"]))

    # ---------------------------------------------------------------- casewise
    case_no = (frame.index.to_numpy() + 1)
    label_var = ds.var(case_labels) if case_labels else None
    if "casewise" in statistics:
        z = diag["zresid"]
        hit = np.ones(len(z), bool) if casewise_all else np.abs(z) >= casewise_sd
        rows = []
        lab = ds.analysis_series(case_labels).loc[frame.index] if case_labels else None
        for k in np.flatnonzero(hit):
            row = [fmt_int(case_no[k])]
            if label_var is not None:
                row.append(value_text(label_var, lab.iloc[k]))
            row += [fmt(z[k]), fmt(ya[k], dvar.decimals), fmt(diag["pred"][k], 5),
                    fmt(diag["resid"][k], 5)]
            rows.append(row)
        if rows:
            section.table(Table("Casewise Diagnosticsᵃ", ["Case Number"]
                                + ([label_var.title] if label_var else [])
                                + ["Std. Residual", dvar.title, "Predicted Value", "Residual"],
                                rows, footnotes=[f"a. {dep_note}"]))
        else:
            section.note("Casewise Diagnostics", "No case has an absolute standardized "
                         f"residual of {fmt(casewise_sd, 1)} or more.")

    # ---------------------------------------------------------------- residuals statistics
    full = "casewise" in statistics or bool(save - {"pred", "zpred", "resid", "zresid"})
    if full or plots or pairs or save or assumptions:
        spec = [("Predicted Value", "pred", dvar.decimals + 4),
                ("Std. Predicted Value", "zpred", 3)]
        if full:
            spec += [("Standard Error of Predicted Value", "sepred", 3),
                     ("Adjusted Predicted Value", "adjpred", dvar.decimals + 4)]
        spec += [("Residual", "resid", dvar.decimals + 5), ("Std. Residual", "zresid", 3)]
        if full:
            spec += [("Stud. Residual", "sresid", 3), ("Deleted Residual", "dresid",
                                                       dvar.decimals + 5),
                     ("Stud. Deleted Residual", "sdresid", 3), ("Mahal. Distance", "mahal", 3),
                     ("Cook's Distance", "cook", 3), ("Centered Leverage Value", "lever", 3)]
        rrows = []
        for label, key, d in spec:
            v = diag[key]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            rrows.append([label, fmt(v.min(), d), fmt(v.max(), d), fmt(v.mean(), d),
                          fmt(v.std(ddof=1), d) if len(v) > 1 else "", fmt_int(len(v))])
        section.table(Table("Residuals Statisticsᵃ", ["", "Minimum", "Maximum", "Mean",
                                                      "Std. Deviation", "N"], rrows,
                            footnotes=[f"a. {dep_note}"] + ([wls_note] if wls else [])))

    # ---------------------------------------------------------------- assumption tests
    if assumptions:
        _assumption_tests(section, assumptions, frame, final.cols, dependent, titles, w, diag,
                          constant, bool(wls))

    # ---------------------------------------------------------------- plots
    zresid = diag["zresid"]
    if "histogram" in plots:
        section.image("Histogram", charts.histogram(
            zresid, f"Histogram — Dependent Variable: {dvar.title}",
            "Regression Standardized Residual"))
    if "pp" in plots:
        section.image("Normal P-P Plot of Regression Standardized Residual", charts.pp_plot(
            zresid, f"Normal P-P Plot — Dependent Variable: {dvar.title}"))
    series = {"DEPENDNT": ya, "*ZPRED": diag["zpred"], "*ZRESID": zresid,
              "*DRESID": diag["dresid"], "*ADJPRED": diag["adjpred"],
              "*SRESID": diag["sresid"], "*SDRESID": diag["sdresid"]}
    for yk, xk in pairs:
        zero = yk in ("*ZRESID", "*SRESID", "*SDRESID", "*DRESID") and \
            xk in ("*ZPRED", "*ZRESID", "*SRESID", "*SDRESID", "*DRESID")
        section.image("Scatterplot", charts.scatter(
            series[xk], series[yk], f"Scatterplot — Dependent Variable: {dvar.title}",
            PLOT_VARS[xk] or dvar.title, PLOT_VARS[yk] or dvar.title, zero_lines=zero))
    if partial_plots and len(final.cols) >= 1:
        sw = np.sqrt(w)
        for c in final.cols:
            others = [o for o in final.cols if o != c]
            base = np.column_stack(([np.ones(n)] if constant else [])
                                   + [frame[o].to_numpy(float) for o in others]) \
                if (others or constant) else np.zeros((n, 0))

            def resid(v):
                if base.shape[1] == 0:
                    return v
                coef = np.linalg.lstsq(base * sw[:, None], v * sw, rcond=None)[0]
                return v - base @ coef
            section.image("Partial Regression Plot", charts.scatter(
                resid(frame[c].to_numpy(float)), resid(ya),
                f"Partial Regression Plot — Dependent Variable: {dvar.title}",
                titles[c], dvar.title, fit_line=True))

    # ---------------------------------------------------------------- save
    if save:
        _save(result, ds, save, ffit, final.cols, data, complete, use, diag, wser, pil,
              constant, dependent)
    return result


def _save(result, ds, save, fit, cols, data, complete, use, diag, wser, level, constant,
          dependent) -> None:
    index = ds.df.index
    xok = data[cols].notna().all(axis=1) & wser.notna() & (wser > 0)
    X = data.loc[xok, cols].to_numpy(float)
    Xd = np.column_stack([np.ones(len(X)), X]) if constant else X
    pred_all = pd.Series(np.nan, index=index)
    pred_all[xok] = Xd @ fit.b
    mse = fit.mse
    G = fit.cov / mse if mse > 0 else fit.cov * 0
    htil = pd.Series(np.nan, index=index)
    htil[xok] = np.einsum("ij,jk,ik->i", Xd, G, Xd)
    w_all = wser.where(xok)
    s = math.sqrt(mse)
    w_use = wser[use].to_numpy(float)
    pu = diag["pred"]
    mp = np.sum(w_use * pu) / w_use.sum()
    sdp = math.sqrt(np.sum(w_use * (pu - mp) ** 2) / (w_use.sum() - 1))
    tq = stats.t.ppf(0.5 + level / 2, max(fit.df_resid, 1))
    sep = s * np.sqrt(htil)

    def case_series(key) -> pd.Series:
        out = pd.Series(np.nan, index=index)
        out[use] = diag[key]
        return out

    yall = numeric(ds, dependent)
    for key in [k for k in SAVE_SPECS if k in save]:
        root, label, _ = SAVE_SPECS[key]
        if key == "pred":
            result.saved[root] = (label, pred_all)
        elif key == "zpred":
            result.saved[root] = (label, (pred_all - mp) / sdp if sdp > 0 else pred_all * 0)
        elif key == "sepred":
            result.saved[root] = (label, sep)
        elif key == "resid":
            result.saved[root] = (label, yall - pred_all)
        elif key == "zresid":
            result.saved[root] = (label, np.sqrt(w_all) * (yall - pred_all) / s)
        elif key == "mcin":
            result.saved["LMCI"] = (f"{pct(level)} L CI for {dependent} mean",
                                    pred_all - tq * sep)
            result.saved["UMCI"] = (f"{pct(level)} U CI for {dependent} mean",
                                    pred_all + tq * sep)
        elif key == "icin":
            ind = np.sqrt(mse / w_all + sep ** 2)
            result.saved["LICI"] = (f"{pct(level)} L CI for {dependent}", pred_all - tq * ind)
            result.saved["UICI"] = (f"{pct(level)} U CI for {dependent}", pred_all + tq * ind)
        elif key in ("dfbeta", "sdfbeta"):
            mat = diag[key]
            for j, name in enumerate(fit.names):
                out = pd.Series(np.nan, index=index)
                out[use] = mat[:, j]
                what = "Intercept" if name == "const" else name
                result.saved[f"{root}{j}"] = (f"{label} {what}", out)
        else:
            result.saved[root] = (label, case_series(key))


def _assumption_tests(section, assumptions, frame, cols, dependent, titles, w, diag,
                      constant, weighted) -> None:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, het_breuschpagan,
                                              het_white)
    from statsmodels.stats.stattools import durbin_watson

    n = len(frame)
    resid = pd.Series(np.sqrt(w) * diag["resid"], index=frame.index)
    X = frame[cols]
    exog = sm.add_constant(X, has_constant="add") if constant else X
    y = frame[dependent]
    model = (sm.WLS(y, exog, weights=w) if weighted else sm.OLS(y, exog)).fit()
    if "normality" in assumptions:
        section.note("Classical Assumption Test 1 — Normality of Residuals",
                     "H0: residuals are normally distributed. Residuals are considered normal "
                     "when Sig. > .05.")
        ks = ks_details(resid.to_numpy())
        column = SimpleNamespace(title="Unstandardized Residual", decimals=3)
        section.table(ks_table([(column, ks)]))
        wv, sp = shapiro(resid.to_numpy())
        verdict = "Normal" if (ks.get("p", 0) > 0.05) else "Not normal"
        section.table(Table("Normality of Residuals — Summary",
                            ["Test", "Statistic", "df", "Sig.", "Decision (α = .05)"],
                            [["Kolmogorov-Smirnov (Lilliefors)", fmt(ks.get("statistic")),
                              fmt_int(n), fmt_p(ks.get("p")) + ("*" if ks.get("lower_bound")
                                                                else ""), verdict],
                             ["Shapiro-Wilk", fmt(wv), fmt_int(n), fmt_p(sp),
                              "Normal" if sp > 0.05 else "Not normal"]],
                            footnotes=["*. This is a lower bound of the true significance."]
                            if ks.get("lower_bound") else []))
    if "multicollinearity" in assumptions and cols:
        M = moments(frame[cols], w if weighted else None)
        tol = tolerances(M, cols)
        rows = [[titles[c], fmt(t), fmt(1 / t),
                 "No multicollinearity" if t > 0.10 and 1 / t < 10 else "Multicollinearity"]
                for c, t in zip(cols, tol)]
        section.note("Classical Assumption Test 2 — Multicollinearity",
                     "No multicollinearity when Tolerance > 0.10 and VIF < 10.")
        section.table(Table("Collinearity Statistics", ["Model", "Tolerance", "VIF",
                                                        "Decision"], rows))
    if "heteroscedasticity" in assumptions and cols:
        section.note("Classical Assumption Test 3 — Heteroscedasticity",
                     "H0: residual variance is constant (homoscedastic). No heteroscedasticity "
                     "when Sig. > .05. Glejser test: |residual| regressed on the predictors.")
        abs_res = resid.abs()
        g = sm.OLS(abs_res, sm.add_constant(X, has_constant="add")).fit()
        grows = []
        for term in ["const"] + cols:
            beta = "" if term == "const" else \
                fmt(g.params[term] * X[term].std(ddof=1) / abs_res.std(ddof=1))
            grows.append(["(Constant)" if term == "const" else titles[term],
                          fmt(g.params[term]), fmt(g.bse[term]), beta, fmt(g.tvalues[term]),
                          fmt_p(g.pvalues[term]),
                          "" if term == "const" else
                          ("No heteroscedasticity" if g.pvalues[term] > 0.05
                           else "Heteroscedasticity")])
        section.table(Table("Glejser Test — Coefficientsᵃ", [
            "Model", ("Unstandardized Coefficients", "B"),
            ("Unstandardized Coefficients", "Std. Error"),
            ("Standardized Coefficients", "Beta"), "t", "Sig.", "Decision"], grows,
            footnotes=["a. Dependent Variable: ABS_RES (absolute unstandardized residual)"]))
        hrows = []
        hex_ = sm.add_constant(X, has_constant="add").to_numpy(float)
        try:
            lm, lm_p, _, _ = het_breuschpagan(resid, hex_)
            hrows.append(["Breusch-Pagan (LM)", fmt(lm), fmt_int(len(cols)), fmt_p(lm_p),
                          "No heteroscedasticity" if lm_p > 0.05 else "Heteroscedasticity"])
        except (ValueError, np.linalg.LinAlgError):
            pass
        try:
            lm, lm_p, _, _ = het_white(resid, hex_)
            k = len(cols)
            hrows.append(["White (LM)", fmt(lm), fmt_int(k * (k + 3) // 2), fmt_p(lm_p),
                          "No heteroscedasticity" if lm_p > 0.05 else "Heteroscedasticity"])
        except (ValueError, AssertionError, np.linalg.LinAlgError):
            pass
        if hrows:
            section.table(Table("Heteroscedasticity Tests", ["Test", "Chi-Square", "df", "Sig.",
                                                             "Decision (α = .05)"], hrows))
    if "autocorrelation" in assumptions:
        section.note("Classical Assumption Test 4 — Autocorrelation",
                     "Durbin-Watson: no autocorrelation when dU < d < 4 − dU (see the "
                     f"Durbin-Watson table for n = {n}, k = {len(cols)}). "
                     "Breusch-Godfrey and Runs test: no autocorrelation when Sig. > .05.")
        arows = [["Durbin-Watson d", fmt(durbin_watson(resid)), "", "", "Compare with dL / dU"]]
        try:
            bg = acorr_breusch_godfrey(model, nlags=1)
            arows.append(["Breusch-Godfrey LM (lag 1)", fmt(bg[0]), "1", fmt_p(bg[1]),
                          "No autocorrelation" if bg[1] > 0.05 else "Autocorrelation"])
        except (ValueError, np.linalg.LinAlgError):
            pass
        rd = runs_details(resid.to_numpy(), "median")
        arows.append(["Runs Test (Z)", fmt(rd["z"]), "", fmt_p(rd["p"]),
                      "No autocorrelation" if rd["p"] > 0.05 else "Autocorrelation"])
        section.table(Table("Autocorrelation Tests", ["Test", "Statistic", "df", "Sig.",
                                                      "Decision (α = .05)"], arows))
        section.table(runs_table([("Unstandardized Residual", rd)], "median"))


def _syntax(dep, blocks, statistics, plots, pairs, save, level, p_enter, p_remove, use_f,
            f_enter, f_remove, constant, missing, selection, wls, casewise_sd, casewise_all,
            partial_plots) -> str:
    kw = {"estimates": "COEFF", "ci": f"CI({level * 100:g})", "bcov": "BCOV",
          "model_fit": "R ANOVA", "r2_change": "CHANGE", "part_partial": "ZPP",
          "collinearity": "COLLIN TOL"}
    lines = ["REGRESSION"]
    if "descriptives" in statistics:
        lines.append("  /DESCRIPTIVES MEAN STDDEV CORR SIG N")
    lines.append(f"  /MISSING {missing.upper().replace('MEAN', 'MEANSUBSTITUTION')}")
    if selection:
        v, rel, val = selection
        lines.append(f"  /SELECT={v} {rel.upper()} {format_code(val)}")
    lines.append("  /STATISTICS " + " ".join(["OUTS"] + [v for k, v in kw.items()
                                                       if k in statistics]))
    crit = f"FIN({f_enter:g}) FOUT({f_remove:g})" if use_f else \
        f"PIN({p_enter:g}) POUT({p_remove:g})"
    lines.append(f"  /CRITERIA={crit}")
    lines.append("  /ORIGIN" if not constant else "  /NOORIGIN")
    if wls:
        lines.append(f"  /REGWGT={wls}")
    lines.append(f"  /DEPENDENT {dep}")
    for m, vs in blocks:
        lines.append(f"  /METHOD={m.upper()} " + " ".join(vs))
    for yk, xk in pairs:
        lines.append(f"  /SCATTERPLOT=({yk} ,{xk})")
    if partial_plots:
        lines.append("  /PARTIALPLOT ALL")
    res = []
    if "durbin_watson" in statistics:
        res.append("DURBIN")
    if "histogram" in plots:
        res.append("HISTOGRAM(ZRESID)")
    if "pp" in plots:
        res.append("NORMPROB(ZRESID)")
    if res:
        lines.append("  /RESIDUALS " + " ".join(res))
    if "casewise" in statistics:
        lines.append("  /CASEWISE PLOT(ZRESID) " + ("ALL" if casewise_all
                                                    else f"OUTLIERS({casewise_sd:g})"))
    if save:
        lines.append("  /SAVE " + " ".join(SAVE_SPECS[k][2] for k in SAVE_SPECS if k in save))
    return "\n".join(lines) + "."

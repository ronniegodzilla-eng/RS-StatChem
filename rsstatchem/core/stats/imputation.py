"""Analyze > Multiple Imputation (SPSS MULTIPLE IMPUTATION) and pooled analysis.

* Analyze Patterns: overall summary (variables / cases / values), variable summary, missing
  value patterns chart and pattern frequencies.
* Impute Missing Data Values: fully conditional specification (FCS, chained equations) or the
  monotone method. Scale variables use Bayesian linear regression (parameters drawn from their
  posterior) or predictive mean matching; categorical variables (nominal / ordinal / string) use
  logistic (multinomial) regression with drawn parameters. Constraints (role, minimum, maximum,
  rounding) as in SPSS. The result is a stacked dataset with ``Imputation_`` = 0 (original
  data), 1..m like SPSS.
* Pooling (Rubin's rules): :func:`rubin` combines estimates and variances; ``pooled_*`` run
  descriptives, one-sample / independent / paired t tests, Pearson correlations (Fisher z),
  linear and binary logistic regression on every imputation and print the SPSS-style table
  with the per-imputation rows and a "Pooled" row (with fraction of missing information,
  relative increase in variance and relative efficiency).

Other procedures can call :func:`is_imputed`, :func:`imputation_views` and :func:`rubin` to add
pooled output of their own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NOMINAL, NUMERIC, ORDINAL, Dataset, Variable
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_df, fmt_int, fmt_p, numeric, value_text
from rsstatchem.core.stats.wp4_util import COLORS, figure, png, sort_key

IMPUTATION_VAR = "Imputation_"


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


# =========================================================================== Analyze Patterns
def _missing_frame(ds: Dataset, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({n: ds.analysis_series(n).isna() for n in names})


def analyze_patterns(ds: Dataset, names: list[str], summary: bool = True,
                     patterns: bool = True, variables: bool = True, min_pct: float = 10.0,
                     max_vars: int = 25) -> Section:
    sec = Section("Multiple Imputation: Analyze Patterns")
    sec.log(f"MULTIPLE IMPUTATION {' '.join(names)}\n  /IMPUTE METHOD=NONE\n  /MISSINGSUMMARIES "
            f"OVERALL VARIABLES (MAXVARS={max_vars} MINPCTMISSING={min_pct:g}) PATTERNS.")
    miss = _missing_frame(ds, names)
    n, p = miss.shape
    if summary:
        parts = [("Variables", int(miss.any().sum()), p),
                 ("Cases", int(miss.any(axis=1).sum()), n),
                 ("Values", int(miss.to_numpy().sum()), n * p)]
        fig = figure(8.4, 3.0)
        rows = []
        for i, (title, inc, tot) in enumerate(parts):
            ax = fig.add_subplot(1, 3, i + 1)
            comp = tot - inc
            if tot:
                ax.pie([comp, inc], colors=["#1f4e9c", "#c8102e"], startangle=90,
                       counterclock=False, wedgeprops=dict(edgecolor="white"),
                       autopct=lambda v: f"{v:.2f}%" if v > 0 else "", textprops=dict(
                           color="white", fontsize=8))
            ax.set_title(title, fontsize=10)
            rows.append([title, fmt_int(comp), _f(100 * comp / tot, 2) if tot else "",
                         fmt_int(inc), _f(100 * inc / tot, 2) if tot else ""])
        fig.legend(["Complete Data", "Incomplete Data"], loc="lower center", ncol=2,
                   frameon=False, fontsize=8)
        sec.image("Overall Summary of Missing Values", png(fig), width=640)
        sec.table(Table("Overall Summary of Missing Values",
                        ["", ("Complete Data", "N"), ("Complete Data", "Percent"),
                         ("Incomplete Data", "N"), ("Incomplete Data", "Percent")], rows))
    if variables:
        pct = 100 * miss.mean()
        order = pct.sort_values(ascending=False)
        rows = []
        for name in order.index[:max_vars]:
            if pct[name] < min_pct:
                continue
            var = ds.var(name)
            vals = numeric(ds, name) if var.type == NUMERIC else None
            rows.append([name, fmt_int(int(miss[name].sum())), _f(pct[name], 1),
                         fmt_int(int((~miss[name]).sum())),
                         _f(vals.mean(), 2) if vals is not None else "",
                         _f(vals.std(ddof=1), 3) if vals is not None else ""])
        sec.table(Table("Variable Summaryᵃ", ["", ("Missing", "N"), ("Missing", "Percent"),
                                              "Valid N", "Mean", "Std. Deviation"], rows,
                        footnotes=[f"a. Maximum number of variables shown: {max_vars}",
                                   f"Minimum percentage of missing values for variable to be "
                                   f"included: {min_pct:.1f}%"]))
    if patterns and p:
        order = list(miss.mean().sort_values().index)  # least missing first, like SPSS
        keys = miss[order].apply(lambda r: tuple(r.to_numpy()), axis=1)
        counts = keys.value_counts()
        pats = sorted(counts.index, key=lambda k: (sum(k), [int(v) for v in k[::-1]]))
        grid = np.array([[1.0 if v else 0.0 for v in k] for k in pats])
        fig = figure(min(max(4.5, 1.0 + 0.45 * len(order)), 9),
                     min(max(3.0, 1.5 + 0.22 * len(pats)), 8))
        ax = fig.add_subplot(111)
        from matplotlib.colors import ListedColormap

        ax.imshow(grid, aspect="auto", cmap=ListedColormap(["#d9e2f3", "#c8102e"]),
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pats)))
        ax.set_yticklabels([str(i + 1) for i in range(len(pats))], fontsize=7)
        ax.set_ylabel("Pattern")
        ax.set_title("Missing Value Patterns (red = missing)", fontsize=10)
        sec.image("Missing Value Patterns", png(fig))
        top = sorted(pats, key=lambda k: -counts[k])[:10]
        fig = figure(6.0, 3.6)
        ax = fig.add_subplot(111)
        pct_cases = [100 * counts[k] / n for k in top]
        ax.bar([str(pats.index(k) + 1) for k in top], pct_cases, color="#1f4e9c")
        ax.set_xlabel("Pattern")
        ax.set_ylabel("Percent")
        ax.set_title("Ten Most Frequently Occurring Patterns")
        sec.image("Pattern Frequencies", png(fig))
        rows = [[str(i + 1), fmt_int(int(counts[k])), _f(100 * counts[k] / n, 1)] +
                ["X" if v else "" for v in k] for i, k in enumerate(pats)]
        sec.table(Table("Missing Value Patterns", ["Pattern", "N", "Percent"] + order, rows,
                        footnotes=["X marks the variables missing in the pattern; variables "
                                   "are ordered from least to most missing."]))
    return sec


# =========================================================================== Imputation
@dataclass
class Constraint:
    role: str = "both"             # both | dependent (impute only) | predictor
    minimum: float | None = None
    maximum: float | None = None
    rounding: float | None = None


@dataclass
class ImputationResult:
    dataset: Dataset
    section: Section
    history: pd.DataFrame | None = None


def _is_categorical(var: Variable) -> bool:
    return var.type != NUMERIC or var.measure in (NOMINAL, ORDINAL)


class _Design:
    """Predictor matrix (dummy coding for categorical predictors) built from current values."""

    def __init__(self, columns: dict, cat: dict, names: list[str]):
        self.columns, self.cat, self.names = columns, cat, names

    def matrix(self, exclude: str, predictors: list[str]) -> np.ndarray:
        parts = [np.ones((len(next(iter(self.columns.values()))), 1))]
        for name in predictors:
            if name == exclude:
                continue
            col = self.columns[name]
            if name in self.cat:
                cats = self.cat[name]
                parts.append(np.column_stack([(col == c).astype(float) for c in cats[1:]])
                             if len(cats) > 1 else np.zeros((len(col), 0)))
            else:
                parts.append(col.astype(float)[:, None])
        return np.hstack(parts)


def _draw_linear(rng, X, y, obs, miss, ridge=1e-5):
    xo, yo = X[obs], y[obs]
    p = X.shape[1]
    xtx = xo.T @ xo + ridge * np.eye(p) * max(np.trace(xo.T @ xo) / p, 1)
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ xo.T @ yo
    resid = yo - xo @ beta
    dof = max(len(yo) - p, 1)
    sigma2 = float(resid @ resid) / rng.chisquare(dof)
    chol = np.linalg.cholesky(xtx_inv * sigma2 + 1e-12 * np.eye(p))
    beta_star = beta + chol @ rng.normal(size=p)
    return beta, beta_star, math.sqrt(sigma2)


def _softmax_fit(X, y_idx, k, ridge=1e-3, iters=25):
    """Multinomial logistic regression (reference = first category) by Newton with a small
    ridge penalty, returning coefficients (p x (k-1)) and their covariance."""
    n, p = X.shape
    beta = np.zeros(p * (k - 1))
    Y = np.zeros((n, k))
    Y[np.arange(n), y_idx] = 1
    for _ in range(iters):
        B = beta.reshape(p, k - 1, order="F")
        eta = np.column_stack([np.zeros(n), X @ B])
        eta -= eta.max(axis=1, keepdims=True)
        P = np.exp(eta)
        P /= P.sum(axis=1, keepdims=True)
        g = (X.T @ (Y[:, 1:] - P[:, 1:])).ravel(order="F") - ridge * beta
        H = np.zeros((p * (k - 1), p * (k - 1)))
        for a in range(k - 1):
            for b in range(k - 1):
                w = P[:, a + 1] * ((a == b) - P[:, b + 1])
                H[a * p:(a + 1) * p, b * p:(b + 1) * p] = -(X.T * w) @ X
        H -= ridge * np.eye(len(beta))
        step = np.linalg.solve(-H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    cov = np.linalg.inv(-H)
    return beta, cov


def _impute_categorical(rng, X, y_codes, obs, miss, k):
    beta, cov = _softmax_fit(X[obs], y_codes[obs].astype(int), k)
    try:
        chol = np.linalg.cholesky(cov + 1e-10 * np.eye(len(beta)))
        beta = beta + chol @ rng.normal(size=len(beta))
    except np.linalg.LinAlgError:
        pass
    B = beta.reshape(X.shape[1], k - 1, order="F")
    eta = np.column_stack([np.zeros(miss.sum()), X[miss] @ B])
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta)
    P /= P.sum(axis=1, keepdims=True)
    u = rng.random(len(P))
    return (P.cumsum(axis=1) < u[:, None]).sum(axis=1).clip(0, k - 1)


def _apply_constraints(rng, values, draw, con: Constraint | None, max_draws: int):
    if con is None:
        return values
    lo = -np.inf if con.minimum is None else con.minimum
    hi = np.inf if con.maximum is None else con.maximum
    out = values.copy()
    for _ in range(max_draws):
        bad = (out < lo) | (out > hi)
        if not bad.any():
            break
        out[bad] = draw(bad.sum())
    out = np.clip(out, lo, hi)
    if con.rounding:
        out = np.round(out / con.rounding) * con.rounding
    return out


def impute(ds: Dataset, names: list[str], m: int = 5, method: str = "auto",
           max_iter: int = 10, scale_model: str = "linear", constraints: dict | None = None,
           seed: int | None = 2000000, max_case_draws: int = 50, donors: int = 5,
           max_pct_missing: float = 100.0, history: bool = False,
           dataset_name: str = "Imputed") -> ImputationResult:
    """SPSS MULTIPLE IMPUTATION /IMPUTE. ``method``: auto | fcs | monotone; ``scale_model``:
    linear | pmm. ``constraints`` maps variable -> :class:`Constraint`."""
    constraints = dict(constraints or {})
    rng = np.random.default_rng(seed)
    sec = Section("Multiple Imputation")
    sec.log(f"MULTIPLE IMPUTATION {' '.join(names)}\n  /IMPUTE METHOD="
            f"{method.upper()} NIMPUTATIONS={m} MAXITER={max_iter}"
            f"{' SCALEMODEL=PMM' if scale_model == 'pmm' else ''}\n"
            f"  /OUTFILE IMPUTATIONS={dataset_name}.")
    n = ds.n_cases
    cat = {}
    raw = {}
    for name in names:
        var = ds.var(name)
        s = ds.analysis_series(name)
        if _is_categorical(var):
            levels = sorted(s.dropna().unique(), key=sort_key)
            cat[name] = levels
            idx = {v: i for i, v in enumerate(levels)}
            raw[name] = np.array([idx.get(v, np.nan) if v == v else np.nan
                                  for v in s.to_numpy(dtype=object)], float)
        else:
            raw[name] = numeric(ds, name).to_numpy(float)
    miss = {nm: np.isnan(raw[nm]) for nm in names}
    pct = {nm: 100 * miss[nm].mean() for nm in names}
    role = {nm: (constraints.get(nm) or Constraint()).role for nm in names}
    targets = [nm for nm in names if miss[nm].any() and role[nm] != "predictor"
               and pct[nm] <= max_pct_missing and (~miss[nm]).sum() >= 2]
    too_many = [nm for nm in names if miss[nm].any() and role[nm] != "predictor"
                and nm not in targets]
    complete_vars = [nm for nm in names if not miss[nm].any()]
    predictors = [nm for nm in names if role[nm] != "dependent"
                  and (nm in targets or not miss[nm].any())]
    for nm in names:   # predictors with missing values that are not imputed cannot be used
        if nm not in targets and miss[nm].any() and nm in predictors:
            predictors.remove(nm)
    order = [nm for nm in names if nm in targets]
    monotone = _is_monotone(miss, order)
    chosen = method
    if method == "auto":
        chosen = "monotone" if monotone else "fcs"
    if chosen == "monotone":
        if not monotone:
            raise ValueError("The data do not have a monotone pattern of missing values; "
                             "use the FCS method.")
        order = sorted(order, key=lambda nm: miss[nm].sum())
    iterations = 1 if chosen == "monotone" else max_iter
    hist_rows = []
    imputed_sets = []
    for imp in range(1, m + 1):
        cur = {nm: raw[nm].copy() for nm in names}
        for nm in order:
            obs_vals = raw[nm][~miss[nm]]
            if chosen == "fcs":
                cur[nm][miss[nm]] = rng.choice(obs_vals, miss[nm].sum())
        design = _Design(cur, cat, names)
        for it in range(1, iterations + 1):
            for idx_var, nm in enumerate(order):
                if chosen == "monotone":
                    preds = [p for p in predictors if p in complete_vars or
                             p in order[:idx_var]]
                else:
                    preds = predictors
                X = design.matrix(nm, preds)
                obs = ~miss[nm]
                mm = miss[nm]
                con = constraints.get(nm)
                if nm in cat:
                    codes = _impute_categorical(rng, X, cur[nm], obs, mm, len(cat[nm]))
                    cur[nm][mm] = codes
                else:
                    y = cur[nm]
                    beta, beta_star, sigma = _draw_linear(rng, X, y, obs, mm)
                    if scale_model == "pmm":
                        pred_obs = X[obs] @ beta
                        pred_mis = X[mm] @ beta_star
                        yo = y[obs]
                        vals = np.empty(mm.sum())
                        for i, pm in enumerate(pred_mis):
                            near = np.argsort(np.abs(pred_obs - pm))[:max(donors, 1)]
                            vals[i] = yo[rng.choice(near)]
                        cur[nm][mm] = vals
                    else:
                        mean = X[mm] @ beta_star
                        vals = mean + rng.normal(size=mm.sum()) * sigma
                        if con is not None:
                            pos = {"i": 0}

                            def draw(k, mean=mean, sigma=sigma):
                                pos["i"] += 1
                                return rng.choice(mean, k) + rng.normal(size=k) * sigma
                            vals = _apply_constraints(rng, vals, draw, con, max_case_draws)
                        cur[nm][mm] = vals
            if history:
                for nm in order:
                    v = cur[nm][miss[nm]]
                    hist_rows.append({"Imputation_": imp, "Iteration": it, "Variable": nm,
                                      "Mean": float(np.mean(v)) if nm not in cat else np.nan,
                                      "SD": float(np.std(v, ddof=1)) if nm not in cat and
                                      len(v) > 1 else np.nan})
        imputed_sets.append(cur)

    # stacked dataset like SPSS: original data (Imputation_ = 0) then imputations 1..m
    frames = []
    base = ds.df.copy()
    for k in range(m + 1):
        f = base.copy()
        if k:
            cur = imputed_sets[k - 1]
            for nm in order:
                col = ds.var(nm).name
                vals = cur[nm]
                if nm in cat:
                    levels = cat[nm]
                    newvals = f[col].to_numpy(dtype=object).copy()
                    for i in np.nonzero(miss[nm])[0]:
                        newvals[i] = levels[int(vals[i])]
                    f[col] = newvals if ds.var(nm).type != NUMERIC else \
                        pd.to_numeric(pd.Series(newvals), errors="coerce").to_numpy()
                else:
                    arr = f[col].to_numpy(float).copy()
                    arr[miss[nm]] = vals[miss[nm]]
                    f[col] = arr
        f.insert(0, IMPUTATION_VAR, float(k))
        frames.append(f)
    stacked = pd.concat(frames, ignore_index=True)
    import copy as _copy

    variables = [Variable(name=IMPUTATION_VAR, type=NUMERIC, decimals=0, measure=NOMINAL,
                          label="Imputation Number", value_labels={0.0: "Original data"},
                          role="split")] + \
        [_copy.deepcopy(v) for v in ds.variables]
    new = Dataset(stacked, variables, name=dataset_name)
    for nm in order:  # user-missing codes were replaced in imputed rows; keep the definitions
        pass
    new.meta["imputation"] = {"m": m, "variables": order, "method": chosen}
    new.modified = True

    # ---- output
    spec = [["Imputation Method", "Fully Conditional Specification" if chosen == "fcs"
             else "Monotone"], ["Number of Imputations", str(m)],
            ["Model for Scale Variables", "Predictive Mean Matching" if scale_model == "pmm"
             else "Linear Regression"],
            ["Interactions Included in Models", "(none)"],
            ["Maximum Percentage of Missing Values", f"{max_pct_missing:.1f}%"],
            ["Maximum Number of Parameters in Imputation Model", "100"]]
    sec.table(Table("Imputation Specifications", ["", ""], spec))
    if constraints:
        rows = []
        for nm in names:
            c = constraints.get(nm) or Constraint()
            rows.append([nm, {"both": "Dependent and Predictor", "dependent": "Dependent",
                              "predictor": "Predictor"}[c.role],
                         _f(c.minimum, 2) if c.minimum is not None else "",
                         _f(c.maximum, 2) if c.maximum is not None else "",
                         _f(c.rounding, 2) if c.rounding else ""])
        sec.table(Table("Imputation Constraints", ["", "Role in Imputation",
                                                   ("Imputed Values", "Minimum"),
                                                   ("Imputed Values", "Maximum"),
                                                   ("Imputed Values", "Rounding")], rows))
    res = [["Imputation Method", "", "Fully Conditional Specification" if chosen == "fcs"
            else "Monotone"]]
    if chosen == "fcs":
        res.append(["Fully Conditional Specification Method Iterations", "", str(max_iter)])
    res += [["Dependent Variables", "Imputed", ", ".join(order) or "(none)"],
            ["", "Not Imputed (Too Many Missing Values)", ", ".join(too_many) or "(none)"],
            ["", "Not Imputed (No Missing Values)", ", ".join(complete_vars) or "(none)"],
            ["Imputation Sequence", "", ", ".join(order + [c for c in complete_vars])]]
    sec.table(Table("Imputation Results", ["", "", ""], res, row_header_cols=2))
    rows = []
    for nm in order:
        if nm in cat:
            mtype = "Logistic Regression"
        else:
            mtype = "Predictive Mean Matching" if scale_model == "pmm" else "Linear Regression"
        effects = [p for p in predictors if p != nm]
        rows.append([nm, mtype, ", ".join(effects) or "(intercept only)",
                     fmt_int(int(miss[nm].sum())), fmt_int(int(miss[nm].sum()) * m)])
    sec.table(Table("Imputation Models", ["", ("Model", "Type"), ("Model", "Effects"),
                                          "Missing Values", "Imputed Values"], rows))
    for nm in order:
        var = ds.var(nm)
        if nm in cat:
            labels = [value_text(var, c) for c in cat[nm]]
            rows = []

            def counts(vals):
                return [int((vals == i).sum()) for i in range(len(labels))]
            obs_counts = counts(raw[nm][~miss[nm]])
            tot = max(sum(obs_counts), 1)
            rows.append(["Original Data", ""] + [f"{c} ({100 * c / tot:.1f}%)"
                                                 for c in obs_counts])
            for k, cur in enumerate(imputed_sets, start=1):
                ic = counts(cur[nm][miss[nm]])
                t = max(sum(ic), 1)
                rows.append(["Imputed Values", str(k)] + [f"{c} ({100 * c / t:.1f}%)" for c in ic])
            for k, cur in enumerate(imputed_sets, start=1):
                cc = counts(cur[nm])
                rows.append(["Complete Data After Imputation", str(k)] +
                            [f"{c} ({100 * c / n:.1f}%)" for c in cc])
            sec.table(Table(nm, ["Data", "Imputation"] + labels, rows, row_header_cols=2,
                            footnotes=["Count (Percent) of each category."]))
        else:
            def stat_row(label, k, vals):
                vals = vals[~np.isnan(vals)]
                return [label, k, fmt_int(len(vals)), _f(vals.mean(), 4) if len(vals) else "",
                        _f(vals.std(ddof=1), 5) if len(vals) > 1 else "",
                        _f(vals.min(), 2) if len(vals) else "",
                        _f(vals.max(), 2) if len(vals) else ""]
            rows = [stat_row("Original Data", "", raw[nm][~miss[nm]])]
            rows += [stat_row("Imputed Values", str(k), cur[nm][miss[nm]])
                     for k, cur in enumerate(imputed_sets, start=1)]
            rows += [stat_row("Complete Data After Imputation", str(k), cur[nm])
                     for k, cur in enumerate(imputed_sets, start=1)]
            sec.table(Table(nm, ["Data", "Imputation", "N", "Mean", "Std. Deviation",
                                 "Minimum", "Maximum"], rows, row_header_cols=2))
    hist = pd.DataFrame(hist_rows) if history else None
    if history and hist is not None and not hist.empty and chosen == "fcs":
        scale_vars = [nm for nm in order if nm not in cat]
        for nm in scale_vars[:6]:
            fig = figure(6.4, 3.4)
            for j, stat in enumerate(("Mean", "SD")):
                ax = fig.add_subplot(1, 2, j + 1)
                for k in range(1, m + 1):
                    h = hist[(hist["Variable"] == nm) & (hist["Imputation_"] == k)]
                    ax.plot(h["Iteration"], h[stat], color=COLORS[(k - 1) % len(COLORS)],
                            lw=1, marker=".", ms=3)
                ax.set_title(f"{stat} of imputed {nm}", fontsize=9)
                ax.set_xlabel("Iteration", fontsize=8)
            sec.image(f"Iteration History: {nm}", png(fig))
        rows = [[str(int(r.Imputation_)), str(int(r.Iteration)), r.Variable, _f(r.Mean, 4),
                 _f(r.SD, 4)] for r in hist.itertuples()]
        sec.table(Table("Iteration History", ["Imputation", "Iteration", "Variable",
                                              "Mean of imputed values",
                                              "Std. Deviation of imputed values"], rows))
    sec.note("", f"The imputed data (original data plus {m} imputations, identified by "
                 f"{IMPUTATION_VAR}) open in a new dataset \"{dataset_name}\". Use Analyze > "
                 f"Multiple Imputation > Pooled Analysis for pooled results; other procedures can "
                 f"be run per imputation with Split File by {IMPUTATION_VAR}.")
    return ImputationResult(new, sec, hist)


def _is_monotone(miss: dict, names: list[str]) -> bool:
    order = sorted(names, key=lambda nm: miss[nm].sum())
    for a, b in zip(order, order[1:]):
        if np.any(miss[a] & ~miss[b]):  # a missing implies b missing
            return False
    return True


# =========================================================================== pooling
def imputation_variable(ds: Dataset) -> str | None:
    """Name of the imputation-number variable: ``Imputation_`` (SPSS), or ``Imputation`` as
    file importers strip the trailing underscore."""
    for name in (IMPUTATION_VAR, "Imputation"):
        if ds.has_name(name) and ds.var(name).type == NUMERIC:
            vals = numeric(ds, name).dropna()
            if len(vals) and (vals >= 0).all() and (vals == vals.round()).all():
                return ds.var(name).name
    return None


def is_imputed(ds: Dataset) -> bool:
    """True when the dataset is a stacked multiple-imputation file (``Imputation_`` with at
    least one imputation > 0)."""
    name = imputation_variable(ds)
    return bool(name and (numeric(ds, name) > 0).any())


def imputation_views(ds: Dataset, include_original: bool = True) -> list[tuple[str, Dataset]]:
    """Split a stacked imputation dataset into ("Original data" | "1".., Dataset) views."""
    vals = numeric(ds, imputation_variable(ds))
    out = []
    for k in sorted(vals.dropna().unique()):
        if k == 0 and not include_original:
            continue
        rows = np.nonzero((vals == k).to_numpy())[0]
        view = Dataset(ds.df.iloc[rows].reset_index(drop=True), list(ds.variables),
                       name=ds.name)
        out.append(("Original data" if k == 0 else fmt_int(k), view))
    return out


@dataclass
class Pooled:
    estimate: np.ndarray
    se: np.ndarray
    t: np.ndarray
    df: np.ndarray
    p: np.ndarray
    fmi: np.ndarray
    riv: np.ndarray
    re: np.ndarray
    within: np.ndarray
    between: np.ndarray
    m: int
    extra: dict = field(default_factory=dict)

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        q = stats.t.ppf(0.5 + level / 2, self.df)
        return self.estimate - q * self.se, self.estimate + q * self.se


def rubin(estimates, variances, df_complete: float | None = None) -> Pooled:
    """Rubin's (1987) rules for m completed-data estimates (m x k) and their sampling variances
    (m x k). df uses Rubin's formula; with ``df_complete`` the Barnard-Rubin (1999) small
    sample adjustment is applied."""
    q = np.atleast_2d(np.asarray(estimates, float))
    u = np.atleast_2d(np.asarray(variances, float))
    m = q.shape[0]
    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    b = q.var(axis=0, ddof=1) if m > 1 else np.zeros_like(qbar)
    t = ubar + (1 + 1 / m) * b
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1 + 1 / m) * b / ubar
        df = np.where(b > 0, (m - 1) * (1 + 1 / r) ** 2, np.inf)
        if df_complete is not None:
            lam = (1 + 1 / m) * b / t
            obs = (df_complete + 1) / (df_complete + 3) * df_complete * (1 - lam)
            df = np.where(b > 0, 1 / (1 / df + 1 / obs), df_complete)
        fmi = np.where(b > 0, (r + 2 / (df + 3)) / (r + 1), 0.0)
        re = 1 / (1 + fmi / m)
        se = np.sqrt(t)
        tstat = qbar / se
    pval = 2 * stats.t.sf(np.abs(tstat), np.where(np.isfinite(df), df, 1e9))
    return Pooled(qbar, se, tstat, df, pval, fmi, np.where(b > 0, r, 0.0), re, ubar, b, m)


def _pool_cols() -> list:
    return ["Fraction Missing Info.", "Relative Increase Variance", "Relative Efficiency"]


def _pool_info(pl: Pooled, i: int) -> list[str]:
    return [_f(pl.fmi[i], 3), _f(pl.riv[i], 3), _f(pl.re[i], 3)]


def _views_or_error(ds: Dataset):
    if not is_imputed(ds):
        raise ValueError(f"The active dataset has no {IMPUTATION_VAR} variable with "
                         "imputations. Run Impute Missing Data Values first.")
    return imputation_views(ds)


def _dfs(df) -> str:
    return fmt_df(float(df)) if np.isfinite(df) else ""


def pooled_descriptives(ds: Dataset, names: list[str]) -> Section:
    views = _views_or_error(ds)
    sec = Section("Descriptives (pooled)")
    sec.log(f"DESCRIPTIVES VARIABLES={' '.join(names)}\n  /STATISTICS=MEAN STDDEV MIN MAX "
            f"SEMEAN.\n* Split by {IMPUTATION_VAR}; pooled with Rubin's rules.")
    rows = []
    for name in names:
        est, var_ = [], []
        for label, view in views:
            x = numeric(view, name).dropna().to_numpy()
            mean = x.mean() if len(x) else np.nan
            se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan
            rows.append([name, label, fmt_int(len(x)), _f(x.min(), 2) if len(x) else "",
                         _f(x.max(), 2) if len(x) else "", _f(mean, 4), _f(se, 5),
                         _f(x.std(ddof=1), 5) if len(x) > 1 else "", "", "", ""])
            if label != "Original data":
                est.append(mean)
                var_.append(se ** 2)
        pl = rubin(np.array(est)[:, None], np.array(var_)[:, None])
        sds = [numeric(v, name).std() for lb, v in views if lb != "Original data"]
        ns = [numeric(v, name).count() for lb, v in views if lb != "Original data"]
        rows.append([name, "Pooled", _f(np.mean(ns), 0), "", "", _f(pl.estimate[0], 4),
                     _f(pl.se[0], 5), _f(np.mean(sds), 5)] + _pool_info(pl, 0))
    sec.table(Table("Descriptive Statistics", ["", "Imputation Number", "N", "Minimum",
                                               "Maximum", "Mean", "Std. Error of Mean",
                                               "Std. Deviation"] + _pool_cols(), rows,
                    row_header_cols=2,
                    footnotes=["Pooled Mean and Std. Error by Rubin's rules; pooled N and Std. "
                               "Deviation are averages over the imputations."]))
    return sec


def pooled_one_sample_t(ds: Dataset, names: list[str], test_value: float = 0.0,
                        level: float = 0.95) -> Section:
    views = _views_or_error(ds)
    sec = Section("T-Test (pooled)")
    sec.log(f"T-TEST /TESTVAL={test_value:g} /VARIABLES={' '.join(names)}.\n"
            f"* Split by {IMPUTATION_VAR}; pooled with Rubin's rules.")
    rows = []
    ci = f"{level * 100:g}% Confidence Interval of the Difference"
    for name in names:
        est, var_ = [], []
        for label, view in views:
            x = numeric(view, name).dropna().to_numpy()
            n = len(x)
            d = x.mean() - test_value
            se = x.std(ddof=1) / math.sqrt(n)
            t = d / se
            p = 2 * stats.t.sf(abs(t), n - 1)
            q = stats.t.ppf(0.5 + level / 2, n - 1)
            rows.append([name, label, _f(t, 3), fmt_int(n - 1), fmt_p(p), _f(d, 5),
                         _f(d - q * se, 5), _f(d + q * se, 5)] + [""] * 3)
            if label != "Original data":
                est.append(d)
                var_.append(se ** 2)
        n = numeric(views[-1][1], name).count()
        pl = rubin(np.array(est)[:, None], np.array(var_)[:, None])
        lo, hi = pl.ci(level)
        rows.append([name, "Pooled", _f(pl.t[0], 3), _dfs(pl.df[0]), fmt_p(pl.p[0]),
                     _f(pl.estimate[0], 5), _f(lo[0], 5), _f(hi[0], 5)] + _pool_info(pl, 0))
        _ = n
    sec.table(Table("One-Sample Test", ["", "Imputation Number", "t", "df", "Sig. (2-tailed)",
                                        "Mean Difference", (ci, "Lower"), (ci, "Upper")] +
                    _pool_cols(), rows, row_header_cols=2,
                    footnotes=[f"Test Value = {test_value:g}"]))
    return sec


def pooled_independent_t(ds: Dataset, names: list[str], group: str, groups: tuple,
                         level: float = 0.95) -> Section:
    views = _views_or_error(ds)
    sec = Section("T-Test (pooled)")
    sec.log(f"T-TEST GROUPS={group}({groups[0]} {groups[1]}) /VARIABLES={' '.join(names)}.\n"
            f"* Split by {IMPUTATION_VAR}; pooled with Rubin's rules.")
    ci = f"{level * 100:g}% Confidence Interval of the Difference"
    rows = []
    for name in names:
        for eq in (True, False):
            est, var_ = [], []
            lab = "Equal variances assumed" if eq else "Equal variances not assumed"
            for label, view in views:
                y = numeric(view, name)
                g = view.analysis_series(group)
                a = y[g == groups[0]].dropna().to_numpy()
                b = y[g == groups[1]].dropna().to_numpy()
                d = a.mean() - b.mean()
                if eq:
                    sp = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / \
                        (len(a) + len(b) - 2)
                    se = math.sqrt(sp * (1 / len(a) + 1 / len(b)))
                    df = len(a) + len(b) - 2
                else:
                    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
                    se = math.sqrt(va + vb)
                    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
                t = d / se
                q = stats.t.ppf(0.5 + level / 2, df)
                rows.append([name, lab, label, _f(t, 3), fmt_df(df),
                             fmt_p(2 * stats.t.sf(abs(t), df)), _f(d, 5), _f(se, 5),
                             _f(d - q * se, 5), _f(d + q * se, 5)] + [""] * 3)
                if label != "Original data":
                    est.append(d)
                    var_.append(se ** 2)
            pl = rubin(np.array(est)[:, None], np.array(var_)[:, None])
            lo, hi = pl.ci(level)
            rows.append([name, lab, "Pooled", _f(pl.t[0], 3), _dfs(pl.df[0]), fmt_p(pl.p[0]),
                         _f(pl.estimate[0], 5), _f(pl.se[0], 5), _f(lo[0], 5), _f(hi[0], 5)] +
                        _pool_info(pl, 0))
    sec.table(Table("Independent Samples Test", ["", "", "Imputation Number", "t", "df",
                                                 "Sig. (2-tailed)", "Mean Difference",
                                                 "Std. Error Difference", (ci, "Lower"),
                                                 (ci, "Upper")] + _pool_cols(), rows,
                    row_header_cols=3))
    return sec


def pooled_paired_t(ds: Dataset, pairs: list[tuple[str, str]], level: float = 0.95) -> Section:
    views = _views_or_error(ds)
    sec = Section("T-Test (pooled)")
    sec.log("T-TEST PAIRS=" + " ".join(a for a, _ in pairs) + " WITH " +
            " ".join(b for _, b in pairs) + f" (PAIRED).\n* Split by {IMPUTATION_VAR}; pooled "
            "with Rubin's rules.")
    ci = f"{level * 100:g}% Confidence Interval of the Difference"
    rows = []
    for a_name, b_name in pairs:
        est, var_ = [], []
        for label, view in views:
            d = (numeric(view, a_name) - numeric(view, b_name)).dropna().to_numpy()
            n = len(d)
            se = d.std(ddof=1) / math.sqrt(n)
            t = d.mean() / se
            q = stats.t.ppf(0.5 + level / 2, n - 1)
            rows.append([f"{a_name} - {b_name}", label, _f(d.mean(), 5), _f(se, 5),
                         _f(d.mean() - q * se, 5), _f(d.mean() + q * se, 5), _f(t, 3),
                         fmt_int(n - 1), fmt_p(2 * stats.t.sf(abs(t), n - 1))] + [""] * 3)
            if label != "Original data":
                est.append(d.mean())
                var_.append(se ** 2)
        pl = rubin(np.array(est)[:, None], np.array(var_)[:, None])
        lo, hi = pl.ci(level)
        rows.append([f"{a_name} - {b_name}", "Pooled", _f(pl.estimate[0], 5), _f(pl.se[0], 5),
                     _f(lo[0], 5), _f(hi[0], 5), _f(pl.t[0], 3), _dfs(pl.df[0]),
                     fmt_p(pl.p[0])] + _pool_info(pl, 0))
    sec.table(Table("Paired Samples Test", ["", "Imputation Number", "Mean",
                                            "Std. Error Mean", (ci, "Lower"), (ci, "Upper"),
                                            "t", "df", "Sig. (2-tailed)"] + _pool_cols(), rows,
                    row_header_cols=2))
    return sec


def pooled_correlations(ds: Dataset, names: list[str]) -> Section:
    """Pearson correlations pooled through Fisher's z (variance 1/(n-3)) with Rubin's rules."""
    views = _views_or_error(ds)
    sec = Section("Correlations (pooled)")
    sec.log(f"CORRELATIONS /VARIABLES={' '.join(names)} /MISSING=PAIRWISE.\n* Split by "
            f"{IMPUTATION_VAR}; pooled through Fisher's z with Rubin's rules.")
    rows = []
    k = len(names)
    for label, view in views + [("Pooled", None)]:
        if view is not None:
            frame = pd.concat({n: numeric(view, n) for n in names}, axis=1)
        for i, a in enumerate(names):
            r_row, p_row, n_row = [], [], []
            for j, b in enumerate(names):
                if i == j:
                    r_row.append("1")
                    p_row.append("")
                    n_row.append(fmt_int(frame[a].count()) if view is not None else "")
                    continue
                if view is not None:
                    pair = frame[[a, b]].dropna()
                    r, p = stats.pearsonr(pair[a], pair[b]) if len(pair) > 2 else (np.nan,
                                                                                   np.nan)
                    r_row.append(_f(r, 3))
                    p_row.append(fmt_p(p))
                    n_row.append(fmt_int(len(pair)))
                else:
                    zs, vs = [], []
                    for lb, v in views:
                        if lb == "Original data":
                            continue
                        pair = pd.concat([numeric(v, a), numeric(v, b)], axis=1).dropna()
                        r = np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]
                        zs.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
                        vs.append(1 / (len(pair) - 3))
                    pl = rubin(np.array(zs)[:, None], np.array(vs)[:, None])
                    r_row.append(_f(np.tanh(pl.estimate[0]), 3))
                    zstat = pl.estimate[0] / pl.se[0]
                    p_row.append(fmt_p(2 * stats.norm.sf(abs(zstat))) if not
                                 np.isfinite(pl.df[0]) else fmt_p(pl.p[0]))
                    n_row.append("")
            rows.append([label, a, "Pearson Correlation"] + r_row)
            rows.append(["", "", "Sig. (2-tailed)"] + p_row)
            if view is not None:
                rows.append(["", "", "N"] + n_row)
    _ = k
    sec.table(Table("Correlations", ["Imputation Number", "", ""] + names, rows,
                    row_header_cols=3,
                    footnotes=["Pooled correlations are back-transformed means of Fisher's z; "
                               "their significance uses Rubin's rules on z."]))
    return sec


def _design(view: Dataset, dep: str, preds: list[str]):
    frame = pd.concat({n: numeric(view, n) for n in [dep] + preds}, axis=1).dropna()
    X = np.column_stack([np.ones(len(frame))] + [frame[p].to_numpy() for p in preds])
    return frame[dep].to_numpy(), X


def pooled_linear(ds: Dataset, dep: str, preds: list[str], level: float = 0.95) -> Section:
    import statsmodels.api as sm

    views = _views_or_error(ds)
    sec = Section("Regression (pooled)")
    sec.log(f"REGRESSION /DEPENDENT {dep} /METHOD=ENTER {' '.join(preds)}.\n* Split by "
            f"{IMPUTATION_VAR}; coefficients pooled with Rubin's rules.")
    names = ["(Constant)"] + preds
    est, var_, rows_ms, rows = [], [], [], []
    dfres = None
    for label, view in views:
        y, X = _design(view, dep, preds)
        fit = sm.OLS(y, X).fit()
        rows_ms.append([label, _f(math.sqrt(max(fit.rsquared, 0)), 3), _f(fit.rsquared, 3),
                        _f(fit.rsquared_adj, 3), _f(math.sqrt(fit.mse_resid), 5)])
        for i, nm in enumerate(names):
            rows.append([label if i == 0 else "", nm, _f(fit.params[i], 3), _f(fit.bse[i], 3),
                         _f(fit.tvalues[i], 3), "", fmt_p(fit.pvalues[i]), "", ""] + [""] * 3)
        if label != "Original data":
            est.append(fit.params)
            var_.append(fit.bse ** 2)
            dfres = fit.df_resid
    pl = rubin(np.array(est), np.array(var_), df_complete=dfres)
    lo, hi = pl.ci(level)
    r2 = [float(r[2]) if r[2] else np.nan for r in rows_ms if r[0] != "Original data"]
    zr = np.arctanh(np.sqrt(np.clip(r2, 0, 0.999999)))
    rows_ms.append(["Pooled", _f(np.tanh(zr.mean()), 3), _f(np.tanh(zr.mean()) ** 2, 3), "",
                    ""])
    sec.table(Table("Model Summary", ["Imputation Number", "R", "R Square",
                                      "Adjusted R Square", "Std. Error of the Estimate"],
                    rows_ms, footnotes=[f"Predictors: (Constant), {', '.join(preds)}",
                                        "Pooled R: back-transformed mean Fisher z of R "
                                        "(Harel, 2009)."]))
    ci = f"{level * 100:g}% Confidence Interval for B"
    for i, nm in enumerate(names):
        rows.append(["Pooled" if i == 0 else "", nm, _f(pl.estimate[i], 3), _f(pl.se[i], 3),
                     _f(pl.t[i], 3), _dfs(pl.df[i]), fmt_p(pl.p[i]), _f(lo[i], 3),
                     _f(hi[i], 3)] + _pool_info(pl, i))
    sec.table(Table("Coefficients", ["Imputation Number", "", "B", "Std. Error", "t", "df",
                                     "Sig.", (ci, "Lower Bound"), (ci, "Upper Bound")] +
                    _pool_cols(), rows, row_header_cols=2,
                    footnotes=[f"Dependent Variable: {dep}",
                               "Pooled df use the Barnard-Rubin small-sample adjustment."]))
    return sec


def pooled_logistic(ds: Dataset, dep: str, preds: list[str], level: float = 0.95) -> Section:
    import statsmodels.api as sm

    views = _views_or_error(ds)
    sec = Section("Logistic Regression (pooled)")
    sec.log(f"LOGISTIC REGRESSION VARIABLES {dep} /METHOD=ENTER {' '.join(preds)}.\n* Split by "
            f"{IMPUTATION_VAR}; coefficients pooled with Rubin's rules.")
    names = preds + ["Constant"]
    est, var_, rows = [], [], []
    for label, view in views:
        y, X = _design(view, dep, preds)
        levels = sorted(np.unique(y))
        if len(levels) != 2:
            raise ValueError(f"{dep} must have exactly two values.")
        yb = (y == levels[1]).astype(float)
        Xo = np.column_stack([X[:, 1:], X[:, 0]])
        fit = sm.Logit(yb, Xo).fit(disp=0)
        for i, nm in enumerate(names):
            wald = (fit.params[i] / fit.bse[i]) ** 2
            rows.append([label if i == 0 else "", nm, _f(fit.params[i], 3), _f(fit.bse[i], 3),
                         _f(wald, 3), "1", fmt_p(fit.pvalues[i]),
                         _f(math.exp(fit.params[i]), 3), "", ""] + [""] * 3)
        if label != "Original data":
            est.append(fit.params)
            var_.append(fit.bse ** 2)
    pl = rubin(np.array(est), np.array(var_))
    lo, hi = pl.ci(level)
    for i, nm in enumerate(names):
        rows.append(["Pooled" if i == 0 else "", nm, _f(pl.estimate[i], 3), _f(pl.se[i], 3),
                     _f(pl.t[i] ** 2, 3), _dfs(pl.df[i]), fmt_p(pl.p[i]),
                     _f(math.exp(pl.estimate[i]), 3), _f(math.exp(lo[i]), 3),
                     _f(math.exp(hi[i]), 3)] + _pool_info(pl, i))
    ci = f"{level * 100:g}% C.I. for EXP(B)"
    sec.table(Table("Variables in the Equation", ["Imputation Number", "", "B", "S.E.", "Wald",
                                                  "df", "Sig.", "Exp(B)", (ci, "Lower"),
                                                  (ci, "Upper")] + _pool_cols(), rows,
                    row_header_cols=2,
                    footnotes=["For the pooled row, Wald = t² and df is Rubin's df of the "
                               "pooled t statistic."]))
    return sec

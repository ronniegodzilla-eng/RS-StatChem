"""Data > Propensity Score Matching, Case Control Matching and Rake Weights (the SPSS built-in
extensions FUZZY / PSM, CASECTRL and SPSSINC RAKE).

* **Propensity Score Matching**: the propensity score is the predicted probability of the
  group indicator from a binary logistic regression on the predictors (SPSS LOGISTIC
  REGRESSION output is shown). Each treated case gets the nearest control whose score lies
  within the match tolerance (|p_t − p_c| ≤ tolerance, optionally on the logit scale or in
  standard deviations of the score, MatchIt's caliper convention), with or without replacement,
  1 or k controls, exact score matches first ("give priority to exact matches") and treated
  cases processed in case order, by descending / ascending score, or in random order (seeded
  MT19937). Saved: the propensity variable, the match ID (ID of the matched case, written on
  both members of a pair) and optionally the number of eligible controls. Balance diagnostics
  report standardized mean differences (pooled-SD denominator for the treated) before and after
  matching, as MatchIt's summary does.
* **Case Control Matching**: controls are matched to each case on key variables, exactly or
  within a tolerance per variable, k controls per case, with or without replacement, drawn at
  random (seed) or in case order; a match-group variable and the matched case's ID are saved.
* **Rake Weights**: iterative proportional fitting (raking) of a weight to marginal control
  totals of one or more categorical variables: w ← w · T_k(j) / Σ_{i∈j} w_i for each variable in
  turn until the largest relative change is below the tolerance; the final weights are scaled
  to the requested total (the sum of the control totals or of the input weights).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, value_text
from rsstatchem.core.stats.ml_common import MLError, measure_of, raw_column, rng, sort_levels

MatchError = MLError


# ============================================================================ greedy matching
def nearest_match(score_t: np.ndarray, score_c: np.ndarray, tolerance: float | None,
                  replace: bool = False, ratio: int = 1, order: str = "data",
                  exact_first: bool = True, seed: int | None = None) -> list[list[int]]:
    """For each treated case the indices of its matched controls (possibly empty).

    Greedy nearest neighbour: treated cases are processed in ``order`` (data / largest /
    smallest / random); each takes the closest available control(s) within ``tolerance``.
    Ties are broken by the lower control index."""
    t = np.asarray(score_t, float)
    c = np.asarray(score_c, float)
    idx = np.arange(len(t))
    if order == "largest":
        idx = idx[np.argsort(-t, kind="mergesort")]
    elif order == "smallest":
        idx = idx[np.argsort(t, kind="mergesort")]
    elif order == "random":
        idx = rng(seed).permutation(len(t))
    avail = np.ones(len(c), bool)
    out: list[list[int]] = [[] for _ in range(len(t))]
    passes = [True, False] if exact_first else [False]
    for want in range(int(ratio)):
        for exact_pass in passes:
            for i in idx:
                if len(out[i]) > want:
                    continue
                d = np.abs(c - t[i])
                ok = avail.copy() if not replace else np.ones(len(c), bool)
                if replace:
                    ok[out[i]] = False
                if exact_pass:
                    ok &= d == 0
                elif tolerance is not None:
                    ok &= d <= tolerance + 1e-12
                if not ok.any():
                    continue
                cand = np.where(ok)[0]
                j = int(cand[np.argmin(d[cand])])
                out[i].append(j)
                if not replace:
                    avail[j] = False
    return out


def smd(x: np.ndarray, treat: np.ndarray, weights: np.ndarray | None = None,
        sd_t: float | None = None) -> float:
    """Standardized mean difference (treated − control) / SD of the treated (MatchIt's default
    for the ATT); ``weights`` give matched-control frequencies."""
    x = np.asarray(x, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    mt = np.average(x[treat], weights=w[treat]) if w[treat].sum() else np.nan
    mc = np.average(x[~treat], weights=w[~treat]) if w[~treat].sum() else np.nan
    s = sd_t if sd_t is not None else np.std(x[treat], ddof=1)
    return float((mt - mc) / s) if s and s > 0 else float("nan")


def propensity_matching(ds: Dataset, group: str, predictors: list[str], propensity_name: str,
                        tolerance: float, case_id: str | None = None,
                        match_id_name: str = "MatchID", eligible_name: str | None = None,
                        replace: bool = False, ratio: int = 1, exact_first: bool = True,
                        order: str = "data", seed: int | None = None, scale: str = "ps",
                        logistic_output: bool = True, balance: bool = True,
                        output_dataset: bool = False) -> Section:
    """Runs inside the editor's undoable mutate: creates the propensity and match variables."""
    from rsstatchem.core.stats.logistic import binary_logistic

    if not predictors:
        raise MatchError("Select at least one predictor.")
    if group in predictors:
        raise MatchError("The group indicator cannot be a predictor.")
    g = pd.to_numeric(raw_column(ds, group), errors="coerce")
    vals = sorted(g.dropna().unique())
    if not set(vals) <= {0.0, 1.0} or len(vals) != 2:
        raise MatchError("The group indicator must be coded 1 (treatment) and 0 (control).")
    for nm in [propensity_name, match_id_name] + ([eligible_name] if eligible_name else []):
        if ds.has_name(nm):
            raise MatchError(f"A variable named {nm} already exists.")
    categorical = {p: ("indicator", "last") for p in predictors
                   if ds.var(p).type != NUMERIC or measure_of(ds, p) == "nominal"}
    res = binary_logistic(ds, group, predictors, categorical=categorical, save=("prob",),
                          hosmer=False)
    section = Section("Propensity Score Matching")
    section.log(f"FUZZY BY={propensity_name} SUPPLIERID={case_id or '$CASENUM'} "
                f"NEWDEMANDERIDVARS={match_id_name} GROUP={group}\n"
                f"  FUZZ={tolerance:g} EXACTPRIORITY={'TRUE' if exact_first else 'FALSE'}\n"
                f"  /OPTIONS SAMPLEWITHREPLACEMENT={'TRUE' if replace else 'FALSE'} "
                f"MINIMIZEMEMORY=TRUE SHUFFLE={'TRUE' if order == 'random' else 'FALSE'}"
                + (f" SEED={seed}" if order == "random" and seed is not None else "") + ".")
    if logistic_output:
        for item in res.section.items:
            section.items.append(item)
    ps = res.saved["PRE"][1].to_numpy(float)
    pv = ds.add_variable(propensity_name, values=ps)
    pv.label, pv.decimals = "Propensity Score", 5
    ok = np.isfinite(ps) & g.notna().to_numpy()
    treat_rows = np.where(ok & (g.to_numpy() == 1))[0]
    ctrl_rows = np.where(ok & (g.to_numpy() == 0))[0]
    if not len(treat_rows) or not len(ctrl_rows):
        raise MatchError("Both groups need cases with valid predictors.")
    if scale == "logit":
        tr = lambda p: np.log(p / (1 - p))  # noqa: E731
    else:
        tr = lambda p: p  # noqa: E731
    st, sc = tr(ps[treat_rows]), tr(ps[ctrl_rows])
    tol = tolerance
    if scale == "sd":
        tol = tolerance * float(np.std(ps[ok], ddof=1))
    matches = nearest_match(st, sc, tol, replace, ratio, order, exact_first, seed)
    ids = (raw_column(ds, case_id).to_numpy(object) if case_id
           else np.arange(1, ds.n_cases + 1).astype(float))
    numeric_id = case_id is None or ds.var(case_id).type == NUMERIC
    mid = np.full(ds.n_cases, np.nan if numeric_id else "", dtype=float if numeric_id
                  else object)
    weights = np.zeros(ds.n_cases)
    n_exact = n_fuzzy = 0
    for i, js in enumerate(matches):
        if not js:
            continue
        r = treat_rows[i]
        mid[r] = ids[ctrl_rows[js[0]]]
        weights[r] = 1
        for j in js:
            c = ctrl_rows[j]
            if (numeric_id and np.isnan(mid[c])) or (not numeric_id and mid[c] == ""):
                mid[c] = ids[r]
            weights[c] += 1 / len(js)
            if st[i] == sc[j]:
                n_exact += 1
            else:
                n_fuzzy += 1
    mv = ds.add_variable(match_id_name, vtype=NUMERIC if numeric_id else "string", values=mid)
    mv.label = "Match ID"
    if numeric_id:
        mv.decimals = 0
    if eligible_name:
        elig = np.full(ds.n_cases, np.nan)
        for i, r in enumerate(treat_rows):
            elig[r] = float((np.abs(sc - st[i]) <= tol + 1e-12).sum())
        ev = ds.add_variable(eligible_name, values=elig)
        ev.label, ev.decimals = "Number of eligible controls", 0
    matched = sum(1 for js in matches if js)
    section.table(Table("Matching Statistics", ["", "Value"], [
        ["Treated cases (group = 1)", fmt_int(len(treat_rows))],
        ["Control cases (group = 0)", fmt_int(len(ctrl_rows))],
        ["Exact matches", fmt_int(n_exact)], ["Fuzzy matches", fmt_int(n_fuzzy)],
        ["Treated cases matched", fmt_int(matched)],
        ["Treated cases unmatched", fmt_int(len(treat_rows) - matched)],
        ["Distinct controls used", fmt_int(int((weights[ctrl_rows] > 0).sum()))],
        ["Match tolerance", fmt(tolerance, 5)
         + {"ps": "", "logit": " (logit scale)", "sd": " (SD of the score)"}[scale]],
        ["Sampling", "with replacement" if replace else "without replacement"],
        ["Controls per case", fmt_int(ratio)]],
        footnotes=["Cases with missing predictor values receive no propensity score and are "
                   "not matched."]))
    if balance:
        treat = np.zeros(ds.n_cases, bool)
        treat[treat_rows] = True
        use = np.zeros(ds.n_cases, bool)
        use[treat_rows] = True
        use[ctrl_rows] = True
        cols = [("Propensity Score", ps)]
        for p in predictors:
            if p in categorical:
                s = raw_column(ds, p)
                for lv in sort_levels(s.dropna().tolist()):
                    cols.append((f"{p} = {value_text(ds.var(p), lv)}",
                                 (s == lv).to_numpy(float)))
            else:
                cols.append((p, raw_column(ds, p).to_numpy(float)))
        rows = []
        mw = weights.copy()
        mt = use & (mw > 0)
        for name, x in cols:
            sub = use & np.isfinite(x)
            tt = treat[sub]
            sd_t = float(np.std(x[sub][tt], ddof=1)) if tt.sum() > 1 else float("nan")
            before = smd(x[sub], tt, None, sd_t)
            subm = mt & np.isfinite(x)
            after = smd(x[subm], treat[subm], mw[subm], sd_t) if subm.sum() else float("nan")
            rows.append([name, fmt(np.mean(x[sub][tt]), 4), fmt(np.mean(x[sub][~tt]), 4),
                         fmt(before, 4),
                         fmt(np.average(x[subm][~treat[subm]], weights=mw[subm][~treat[subm]])
                             if (subm & ~treat).any() else np.nan, 4), fmt(after, 4)])
        section.table(Table("Balance Diagnostics", [
            "", ("All Cases", "Mean Treated"), ("All Cases", "Mean Control"),
            ("All Cases", "Std. Mean Diff."), ("Matched", "Mean Control"),
            ("Matched", "Std. Mean Diff.")], rows,
            footnotes=["Standardized mean difference = (mean treated − mean control) / SD of "
                       "the treated cases; matched controls are weighted by 1/k when a "
                       "treated case has k controls (and by their reuse count with "
                       "replacement). |SMD| < 0.1 indicates good balance."]))
    section.result = {"matches": matches, "treat_rows": treat_rows,  # type: ignore
                      "ctrl_rows": ctrl_rows, "ps": ps, "weights": weights}
    return section


# ============================================================================ case control
def case_control_matching(ds: Dataset, group: str, keys: list[str], tolerances: dict | None,
                          case_id: str | None = None, match_id_name: str = "MatchID",
                          group_name: str = "MatchGroup", ratio: int = 1,
                          replace: bool = False, randomize: bool = True,
                          seed: int | None = None) -> Section:
    """CASECTRL: cases (group = 1) are matched to controls (group = 0) on ``keys``; a key
    with tolerance 0 (or a string key) must match exactly."""
    if not keys:
        raise MatchError("Select at least one key variable.")
    g = pd.to_numeric(raw_column(ds, group), errors="coerce").to_numpy()
    if not set(np.unique(g[np.isfinite(g)])) <= {0.0, 1.0}:
        raise MatchError("The group variable must be coded 1 (case) and 0 (control).")
    for nm in (match_id_name, group_name):
        if ds.has_name(nm):
            raise MatchError(f"A variable named {nm} already exists.")
    tolerances = dict(tolerances or {})
    cols = []
    ok = np.isfinite(g)
    for k in keys:
        s = raw_column(ds, k)
        ok &= s.notna().to_numpy()
        cols.append((k, s.to_numpy(object), ds.var(k).type == NUMERIC))
    cases = np.where(ok & (g == 1))[0]
    ctrls = np.where(ok & (g == 0))[0]
    gen = rng(seed)
    order = gen.permutation(cases) if randomize else cases
    avail = np.ones(ds.n_cases, bool)
    ids = (raw_column(ds, case_id).to_numpy(object) if case_id
           else np.arange(1, ds.n_cases + 1).astype(float))
    numeric_id = case_id is None or ds.var(case_id).type == NUMERIC
    mid = np.full(ds.n_cases, np.nan if numeric_id else "", dtype=float if numeric_id
                  else object)
    grp = np.full(ds.n_cases, np.nan)
    tried = matched_cases = n_ctrl = 0
    per_case = []
    for gi, r in enumerate(order, 1):
        tried += 1
        elig = np.ones(len(ctrls), bool)
        for k, vals, num in cols:
            tol = float(tolerances.get(k, 0) or 0)
            if num and tol > 0:
                elig &= np.abs(vals[ctrls].astype(float) - float(vals[r])) <= tol + 1e-12
            else:
                elig &= vals[ctrls] == vals[r]
        if not replace:
            elig &= avail[ctrls]
        cand = ctrls[elig]
        if randomize and len(cand) > ratio:
            cand = gen.choice(cand, int(ratio), replace=False)
        else:
            cand = cand[:int(ratio)]
        per_case.append(len(cand))
        if len(cand):
            matched_cases += 1
            grp[r] = gi
            mid[r] = ids[cand[0]]
            for c in cand:
                grp[c] = gi if np.isnan(grp[c]) else grp[c]
                if (numeric_id and np.isnan(mid[c])) or (not numeric_id and mid[c] == ""):
                    mid[c] = ids[r]
                avail[c] = False
                n_ctrl += 1
    mv = ds.add_variable(match_id_name, vtype=NUMERIC if numeric_id else "string", values=mid)
    mv.label = "ID of the matched case"
    gv = ds.add_variable(group_name, values=grp)
    gv.label, gv.decimals = "Match group", 0
    section = Section("Case Control Matching")
    section.log(f"CASECTRL STRATAVAR={group} CASECTRLVAR={group} "
                f"MATCHVARS={' '.join(keys)}\n  /OPTIONS MATCHGROUPVAR={group_name} "
                f"DRAWRANDOM={'YES' if randomize else 'NO'} "
                f"SAMPLEWITHREPLACEMENT={'YES' if replace else 'NO'} MAXCONTROLS={ratio}"
                + (f" SEED={seed}" if randomize and seed is not None else "") + ".")
    section.table(Table("Match Summary", ["", "Value"], [
        ["Cases tried", fmt_int(tried)], ["Cases matched", fmt_int(matched_cases)],
        ["Cases with no match", fmt_int(tried - matched_cases)],
        ["Controls drawn", fmt_int(n_ctrl)],
        ["Cases with fewer than the requested controls",
         fmt_int(sum(1 for k in per_case if 0 < k < ratio))],
        ["Available controls", fmt_int(len(ctrls))]]))
    tol_rows = [[k, "exact" if not float(tolerances.get(k, 0) or 0) else
                 fmt(float(tolerances[k]), 4)] for k in keys]
    section.table(Table("Match Keys", ["Key", "Tolerance"], tol_rows))
    section.result = {"group": grp, "match": mid}  # type: ignore[attr-defined]
    return section


# ============================================================================ raking
def rake(cats: list[np.ndarray], targets: list[dict], w0: np.ndarray | None = None,
         max_iter: int = 1000, tol: float = 1e-8) -> tuple[np.ndarray, int, float]:
    """Iterative proportional fitting. ``cats[k]`` holds category codes, ``targets[k]`` maps a
    code to its control total. Returns weights, iterations and the last relative change."""
    n = len(cats[0])
    w = np.ones(n) if w0 is None else np.asarray(w0, float).copy()
    change = math.inf
    for it in range(1, int(max_iter) + 1):
        old = w.copy()
        for c, t in zip(cats, targets):
            for code, tot in t.items():
                sel = c == code
                s = w[sel].sum()
                if s > 0:
                    w[sel] *= tot / s
        change = float(np.max(np.abs(w - old) / np.maximum(np.abs(old), 1e-300)))
        if change < tol:
            return w, it, change
    return w, int(max_iter), change


def rake_weights(ds: Dataset, specs: list[tuple[str, dict]], weight_name: str = "rakewt",
                 initial: str | None = None, total: float | None = None,
                 proportions: bool = True, max_iter: int = 1000, tol: float = 1e-6,
                 trim: tuple[float, float] | None = None) -> Section:
    """``specs`` = [(variable, {value: target})]; targets are proportions (normalised) when
    ``proportions`` else population counts. ``total`` is the weighted total of the result
    (default: the sum of the counts, or the sum of the initial weights for proportions)."""
    if not specs:
        raise MatchError("Specify at least one control variable with its targets.")
    if ds.has_name(weight_name):
        raise MatchError(f"A variable named {weight_name} already exists.")
    ok = np.ones(ds.n_cases, bool)
    w0 = None
    if initial:
        w0 = pd.to_numeric(raw_column(ds, initial), errors="coerce").to_numpy()
        ok &= np.isfinite(w0) & (w0 > 0)
    cols = []
    for v, t in specs:
        s = raw_column(ds, v)
        ok &= s.notna().to_numpy()
        cols.append(s.to_numpy(object))
    rows = np.where(ok)[0]
    if not len(rows):
        raise MatchError("No valid cases.")
    base = (w0[rows] if w0 is not None else np.ones(len(rows)))
    if proportions:
        tot = float(total) if total else float(base.sum())
    else:
        tot = float(total) if total else float(sum(specs[0][1].values()))
    cats, tg = [], []
    for (v, t), col in zip(specs, cols):
        sub = col[rows]
        keys = list(t)
        present = set(sub.tolist())
        missing = [k for k in keys if k not in present]
        if missing:
            raise MatchError(f"{v}: no cases have the target value(s) "
                             + ", ".join(str(m) for m in missing) + ".")
        extra = present - set(keys)
        if extra:
            raise MatchError(f"{v}: targets are missing for the value(s) "
                             + ", ".join(str(e) for e in sorted(extra, key=str)) + ".")
        s = float(sum(t.values()))
        if s <= 0:
            raise MatchError(f"{v}: the targets must be positive.")
        tg.append({k: tot * float(x) / s for k, x in t.items()})
        cats.append(sub)
    w, it, change = rake(cats, tg, base, max_iter, tol)
    trimmed = 0
    if trim:
        lo, hi = trim
        for _ in range(50):
            mean = w.mean()
            low, high = w < lo * mean, w > hi * mean
            if not (low.any() or high.any()):
                break
            trimmed = int(low.sum() + high.sum())
            w = np.clip(w, lo * mean, hi * mean)
            w, it2, change = rake(cats, tg, w, max_iter, tol)
            it += it2
    col = np.full(ds.n_cases, np.nan)
    col[rows] = w
    var = ds.add_variable(weight_name, values=col)
    var.label, var.decimals = "Raked weight", 4
    section = Section("Rake Weights")
    section.log("SPSSINC RAKE " + " ".join(
        f"DIM{k + 1}={v} " + " ".join(f"{val} {x}" for val, x in t.items())
        for k, (v, t) in enumerate(specs)) + f"\n  FINALWEIGHT={weight_name}"
        + (f" /OPTIONS ITERATIONS={max_iter} CONVERGENCE={tol:g}"))
    rows_m = []
    for (v, t), sub, tgt in zip(specs, cats, tg):
        for k in t:
            sel = sub == k
            rows_m.append([v, value_text(ds.var(v), k), fmt_int(int(sel.sum())),
                           fmt(base[sel].sum() / base.sum() * 100, 2),
                           fmt(tgt[k] / tot * 100, 2), fmt(w[sel].sum() / w.sum() * 100, 2),
                           fmt(w[sel].sum(), 2)])
    section.table(Table("Control Totals", ["Variable", "Category", "Unweighted N",
                                           "Input %", "Target %", "Raked %",
                                           "Raked Total"], rows_m, row_header_cols=2))
    cv = float(w.std(ddof=0) / w.mean()) if w.mean() else float("nan")
    section.table(Table("Weight Summary", ["", "Value"], [
        ["Cases weighted", fmt_int(len(rows))], ["Sum of weights", fmt(w.sum(), 3)],
        ["Minimum", fmt(w.min(), 4)], ["Maximum", fmt(w.max(), 4)],
        ["Mean", fmt(w.mean(), 4)], ["Coefficient of variation", fmt(cv, 4)],
        ["Design effect (Kish, 1 + CV²)", fmt(1 + cv ** 2, 4)],
        ["Iterations", fmt_int(it)], ["Largest relative change at the end", fmt(change, 8)]]
        + ([["Weights trimmed", fmt_int(trimmed)]] if trim else []),
        footnotes=["Converged." if change < tol else
                   "Raking did not converge within the iteration limit."]))
    section.result = {"weights": col, "iterations": it}  # type: ignore[attr-defined]
    return section


def parse_targets(text: str, numeric: bool = True) -> dict:
    """"1=0.4; 2=0.6" (or "1 0.4 2 0.6") -> {1.0: 0.4, 2.0: 0.6}."""
    out = {}
    text = str(text or "").strip()
    if "=" in text or ";" in text:
        for part in text.replace("\n", ";").split(";"):
            if not part.strip():
                continue
            k, _, v = part.partition("=")
            key = float(k.strip().replace(",", ".")) if numeric else k.strip()
            out[key] = float(v.strip().replace(",", "."))
    else:
        toks = text.split()
        for k, v in zip(toks[0::2], toks[1::2]):
            out[float(k.replace(",", ".")) if numeric else k] = float(v.replace(",", "."))
    return out


def selftest() -> None:
    m = nearest_match(np.array([0.5, 0.2]), np.array([0.21, 0.52, 0.9]), 0.05)
    assert m == [[1], [0]]
    cats = [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])]
    w, _, _ = rake(cats, [{0: 3.0, 1: 1.0}, {0: 2.0, 1: 2.0}])
    assert abs(w[:2].sum() - 3) < 1e-8 and abs(w[[0, 2]].sum() - 2) < 1e-8

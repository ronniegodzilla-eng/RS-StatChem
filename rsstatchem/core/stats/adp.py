"""Transform > Prepare Data for Modeling > Automatic (SPSS ADP, Data Preparation add-on),
following the main steps of IBM's "Automated Data Preparation Algorithms" chapter:

1. **Date handling**: date fields become year, month and day fields plus the elapsed time to a
   reference date (best unit: days < 31, months = days/30.4375 < 366 days, else years =
   days/365.25); the date field itself is excluded.
2. **Screening**: fields with more than 50% missing, constant continuous fields, categorical
   fields whose mode holds more than 95% of the cases and nominal fields with more than 100
   categories are excluded.
3. **Measurement recasting**: continuous fields with fewer than 5 distinct values become
   ordinal; ordinal fields with more than 10 categories become continuous.
4. **Outliers** (continuous): values more than ``cutoff`` (3) robust standard deviations from
   the robust mean are trimmed to the cutoff (or set to missing). The robust mean / SD are
   computed iteratively without the flagged values (an approximation of IBM's
   interval-based estimate that also assumes at most a few percent outliers).
5. **Missing values**: continuous → mean, ordinal → median, nominal → mode.
6. **Categorical handling**: nominal categories reordered from least to most frequent
   (codes 0, 1, …); sparse categories merged — supervised (with a target: adjacent / any pair
   with the largest p-value > α = 0.05 merged, a one-level CHAID with F or chi-square tests) or
   unsupervised (categories with fewer than 10% of the cases merged with their neighbour).
7. **Continuous handling**: with a continuous target, predictors whose correlation p-value
   exceeds 0.05 are dropped (feature selection); continuous predictors are rescaled by z-score
   (mean 0, SD 1) or min–max (0–100).
8. **Target**: a continuous target gets the Box–Cox transformation (λ from a grid −3 … 3 by
   0.5, maximising the profile log-likelihood) followed by the z-score; a nominal target is
   reordered.

Prepared fields are saved with the suffix "_transformed" (SPSS default). Not implemented:
feature construction (principal components of correlated predictors), supervised binning of
continuous predictors for categorical targets and the interactive model viewer.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import STRING, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import MLError, measure_of, raw_column, sort_levels

AdpError = MLError
DEFAULTS = {"max_missing": 50.0, "max_mode": 95.0, "max_categories": 100,
            "cont_to_ord": 5, "ord_to_cont": 10, "outliers": "trim", "cutoff": 3.0,
            "impute": True, "reorder_nominal": True, "merge": True, "alpha": 0.05,
            "sparse_pct": 10.0, "rescale": "zscore", "select": True, "select_p": 0.05,
            "boxcox": True, "dates": True, "reference": None, "suffix": "_transformed"}


def robust_mean_sd(x: np.ndarray, cutoff: float = 3.0, max_iter: int = 50):
    x = x[np.isfinite(x)]
    keep = np.ones(len(x), bool)
    m, s = x.mean(), x.std(ddof=1)
    for _ in range(max_iter):
        new = np.abs(x - m) <= cutoff * s if s > 0 else np.ones(len(x), bool)
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
        m, s = x[keep].mean(), x[keep].std(ddof=1)
    return float(m), float(s)


def boxcox_grid(y: np.ndarray, lo: float = -3.0, hi: float = 3.0, step: float = 0.5):
    """λ maximising the Box–Cox profile log-likelihood over the grid (ties: smallest λ)."""
    y = np.asarray(y, float)
    c = 0.0 if y.min() > 0 else 1.0 - y.min()
    yy = y + c
    n = len(yy)
    best = None
    for lam in np.arange(lo, hi + 1e-9, step):
        z = np.log(yy) if abs(lam) < 1e-12 else (yy ** lam - 1) / lam
        ll = -n / 2 * np.log(z.var()) + (lam - 1) * np.log(yy).sum()
        if best is None or ll > best[1] + 1e-12:
            best = (float(lam), ll, z)
    return best[0], best[2], c


def _merge_supervised(codes: np.ndarray, y: np.ndarray, ordinal: bool, target_kind: str,
                      alpha: float) -> list[list[int]]:
    """One-level CHAID merging; returns groups of original category codes."""
    cats = sorted(set(codes[codes >= 0].tolist()))
    groups = [[c] for c in cats]

    def p_pair(a, b):
        ia = np.isin(codes, a)
        ib = np.isin(codes, b)
        if target_kind == "scale":
            ya, yb = y[ia], y[ib]
            if len(ya) < 2 or len(yb) < 2:
                return 1.0
            return float(stats.f_oneway(ya, yb).pvalue)
        levels = sorted(set(y[ia | ib].tolist()))
        if len(levels) < 2:
            return 1.0
        tab = np.array([[np.sum(ia & (y == lv)), np.sum(ib & (y == lv))] for lv in levels])
        try:
            return float(stats.chi2_contingency(tab, correction=False)[1])
        except ValueError:
            return 1.0

    while len(groups) > 1:
        pairs = [(i, i + 1) for i in range(len(groups) - 1)] if ordinal else \
            [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
        ps = [p_pair(groups[i], groups[j]) for i, j in pairs]
        k = int(np.argmax(ps))
        if ps[k] <= alpha:
            break
        i, j = pairs[k]
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return groups


def _merge_unsupervised(codes: np.ndarray, pct: float) -> list[list[int]]:
    cats = sorted(set(codes[codes >= 0].tolist()))
    counts = {c: int((codes == c).sum()) for c in cats}
    total = sum(counts.values())
    thr = total * pct / 100
    groups, cur = [], []
    for c in cats:
        cur.append(c)
        if sum(counts[x] for x in cur) >= thr:
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            groups[-1] += cur
        else:
            groups.append(cur)
    if len(groups) < 2 and len(cats) >= 2:     # keep at least two categories (IBM)
        groups = [cats[:-1], cats[-1:]]
    return groups


def prepare_data(ds: Dataset, inputs: list[str], target: str | None = None,
                 settings: dict | None = None, save: bool = True) -> Section:
    s = {**DEFAULTS, **(settings or {})}
    if not inputs:
        raise AdpError("Select at least one input field.")
    if target in inputs:
        raise AdpError("The target cannot also be an input.")
    n = ds.n_cases
    section = Section("Automated Data Preparation")
    section.log("ADP\n  /FIELDS " + (f"TARGET={target} " if target else "")
                + f"INPUT={' '.join(inputs)}\n  /PREPDATETIME DATEDURATION=YES\n"
                f"  /SCREENING PCTMISSING=YES({s['max_missing']:g}) "
                f"UNIQUECAT=YES({s['max_categories']}) SINGLECAT=YES({s['max_mode']:g})\n"
                f"  /ADJUSTLEVEL INPUT=YES MAXVALORDINAL={s['ord_to_cont']} "
                f"MINVALCONTINUOUS={s['cont_to_ord']}\n"
                f"  /OUTLIERHANDLING INPUT=YES CUTOFF=SD({s['cutoff']:g}) "
                f"REPLACEWITH={'CUTOFFVALUE' if s['outliers'] == 'trim' else 'MISSING'}\n"
                f"  /REPLACEMISSING INPUT={'YES' if s['impute'] else 'NO'}\n"
                f"  /RESCALE INPUT={s['rescale'].upper()} TARGET="
                f"{'BOXCOX' if s['boxcox'] else 'NONE'}\n"
                f"  /TRANSFORM MERGESUPERVISED={'YES' if s['merge'] and target else 'NO'} "
                f"MERGEUNSUPERVISED={'YES' if s['merge'] and not target else 'NO'}\n"
                f"  /OUTFILE PREPXML=* SUFFIX={s['suffix']}.")
    actions: list[list[str]] = []
    out_cols: dict[str, tuple[np.ndarray, str, str]] = {}   # name -> (values, measure, label)
    excluded: list[tuple[str, str]] = []
    y = None
    target_kind = None
    if target:
        target_kind = measure_of(ds, target)
        y = raw_column(ds, target)
    ref = pd.Timestamp(s["reference"] or _dt.date.today())
    fields = []
    for f in inputs:
        v = ds.var(f)
        if v.type == "date" and s["dates"]:
            d = pd.to_datetime(ds.df[f], errors="coerce")
            days = (ref - d).dt.days.to_numpy(float)
            mn = np.nanmin(np.abs(days)) if np.isfinite(days).any() else 0
            unit, div = ("days", 1.0) if mn < 31 else ("months", 30.4375) if mn < 366 else \
                ("years", 365.25)
            for part, vals in (("year", d.dt.year), ("month", d.dt.month), ("day", d.dt.day)):
                fields.append((f"{f}_{part}", vals.to_numpy(float), "ordinal", f"{f} {part}"))
            fields.append((f"{f}_{unit}", days / div, "scale", f"{f}: {unit} since reference"))
            actions.append([f, "Date", f"Extracted year, month, day and elapsed {unit}"])
            continue
        if v.type == STRING:
            col = raw_column(ds, f).to_numpy(object)
        else:
            col = pd.to_numeric(raw_column(ds, f), errors="coerce").to_numpy(float)
        fields.append((f, col, measure_of(ds, f) if v.type != STRING else "nominal", v.title))
    sel_p = {}
    for name, col, kind, label in fields:
        acts = []
        miss = pd.isna(col) if col.dtype == object else ~np.isfinite(col)
        pm = 100 * miss.mean() if n else 0
        if pm > s["max_missing"]:
            excluded.append((name, f"{pm:.1f}% missing"))
            continue
        if col.dtype == object or kind != "scale":
            vals = pd.Series(col)[~miss]
            vc = vals.value_counts()
            if len(vc) == 0:
                excluded.append((name, "no valid values"))
                continue
            if 100 * vc.iloc[0] / len(vals) > s["max_mode"]:
                excluded.append((name, f"mode holds {100 * vc.iloc[0] / len(vals):.1f}% of "
                                       "cases"))
                continue
            if kind == "nominal" and len(vc) > s["max_categories"]:
                excluded.append((name, f"{len(vc)} categories"))
                continue
        else:
            x = col[~miss]
            if len(x) == 0 or x.max() == x.min():
                excluded.append((name, "constant"))
                continue
        # ---- measurement recasting
        if kind == "scale" and col.dtype != object:
            nd = len(np.unique(col[~miss]))
            if nd < s["cont_to_ord"]:
                kind = "ordinal"
                acts.append(f"Continuous recast as ordinal ({nd} values)")
        elif kind == "ordinal" and col.dtype != object:
            nd = len(np.unique(col[~miss]))
            if nd > s["ord_to_cont"]:
                kind = "scale"
                acts.append(f"Ordinal recast as continuous ({nd} values)")
        if kind == "scale":
            x = col.astype(float).copy()
            if s["outliers"] != "none":
                m, sd = robust_mean_sd(x, s["cutoff"])
                lo, hi = m - s["cutoff"] * sd, m + s["cutoff"] * sd
                out = np.isfinite(x) & ((x < lo) | (x > hi))
                if out.any():
                    if s["outliers"] == "trim":
                        x = np.where(x < lo, lo, np.where(x > hi, hi, x))
                        acts.append(f"{int(out.sum())} outlier(s) trimmed to "
                                    f"[{fmt(lo, 3)}, {fmt(hi, 3)}]")
                    else:
                        x = np.where(out, np.nan, x)
                        acts.append(f"{int(out.sum())} outlier(s) set to missing")
            if s["impute"] and (~np.isfinite(x)).any():
                k = int((~np.isfinite(x)).sum())
                x = np.where(np.isfinite(x), x, np.nanmean(x))
                acts.append(f"{k} missing value(s) replaced by the mean")
            if target and target_kind == "scale" and s["select"]:
                yy = pd.to_numeric(y, errors="coerce").to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(yy)
                if ok.sum() > 3 and np.std(x[ok]) > 0:
                    r, p = stats.pearsonr(x[ok], yy[ok])
                    sel_p[name] = p
                    if p > s["select_p"]:
                        excluded.append((name, f"not related to the target (p = {fmt(p, 3)})"))
                        continue
            if s["rescale"] == "zscore":
                sd = np.nanstd(x, ddof=1)
                x = (x - np.nanmean(x)) / (sd if sd > 0 else 1)
                acts.append("Rescaled (z-score)")
            elif s["rescale"] == "minmax":
                lo, hi = np.nanmin(x), np.nanmax(x)
                x = (x - lo) / (hi - lo) * 100 if hi > lo else x * 0
                acts.append("Rescaled (min-max 0-100)")
            out_cols[name + s["suffix"]] = (x, "scale", f"{label} (prepared)")
        else:
            ser = pd.Series(col)
            if s["impute"] and ser.isna().any():
                k = int(ser.isna().sum())
                if kind == "ordinal" and ser.dtype != object:
                    fill = float(np.nanmedian(ser.to_numpy(float)))
                    acts.append(f"{k} missing value(s) replaced by the median")
                else:
                    vc = ser.dropna().value_counts()
                    fill = sorted(vc[vc == vc.max()].index, key=lambda v: (isinstance(v, str),
                                                                            v))[0]
                    acts.append(f"{k} missing value(s) replaced by the mode")
                ser = ser.where(ser.notna(), fill)
            levels = sort_levels(ser.dropna().tolist())
            if kind == "nominal" and s["reorder_nominal"]:
                vc = ser.value_counts()
                levels = sorted(levels, key=lambda v: (vc.get(v, 0), isinstance(v, str), v))
                acts.append("Categories reordered from least to most frequent")
            codes = np.array([levels.index(v) if pd.notna(v) else -1 for v in ser])
            if s["merge"] and len(levels) > 2:
                if target is not None:
                    yy = y.to_numpy(object) if target_kind != "scale" else \
                        pd.to_numeric(y, errors="coerce").to_numpy(float)
                    okm = codes >= 0
                    ok_y = pd.notna(pd.Series(yy)).to_numpy()
                    groups = _merge_supervised(np.where(okm & ok_y, codes, -1),
                                               np.array(yy), kind == "ordinal",
                                               "scale" if target_kind == "scale" else "cat",
                                               s["alpha"])
                else:
                    groups = _merge_unsupervised(codes, s["sparse_pct"])
                if len(groups) < len(levels):
                    acts.append(f"{len(levels)} categories merged into {len(groups)}")
                    mapping = {c: gi for gi, g in enumerate(groups) for c in g}
                    codes = np.array([mapping.get(c, -1) if c >= 0 else -1 for c in codes])
            vals = np.where(codes >= 0, codes, np.nan).astype(float)
            out_cols[name + s["suffix"]] = (vals, kind, f"{label} (prepared)")
        actions.append([name, {"scale": "Continuous", "ordinal": "Ordinal",
                               "nominal": "Nominal"}[kind], "; ".join(acts) or "None"])
    if target:
        ty = y
        if target_kind == "scale" and s["boxcox"]:
            yy = pd.to_numeric(ty, errors="coerce").to_numpy(float)
            ok = np.isfinite(yy)
            lam, z, c = boxcox_grid(yy[ok])
            zz = (z - z.mean()) / z.std(ddof=1)
            vals = np.full(n, np.nan)
            vals[ok] = zz
            out_cols[target + s["suffix"]] = (vals, "scale", f"{ds.var(target).title} "
                                                             "(prepared)")
            actions.append([target + " (target)", "Continuous",
                            f"Box-Cox (lambda = {lam:g}, shift {c:g}) and z-score"])
        elif target_kind == "nominal":
            vc = ty.value_counts()
            levels = sorted(sort_levels(ty.dropna().tolist()),
                            key=lambda v: (vc.get(v, 0), isinstance(v, str), v))
            vals = np.array([levels.index(v) if pd.notna(v) else np.nan for v in ty], float)
            out_cols[target + s["suffix"]] = (vals, "nominal", f"{ds.var(target).title} "
                                                               "(prepared)")
            actions.append([target + " (target)", "Nominal",
                            "Categories reordered from least to most frequent"])
    if not any(k != (target or "") + s["suffix"] for k in out_cols):
        raise AdpError("All input fields were excluded by the screening.")
    section.table(Table("Field Processing Summary", ["", "Number of Fields"], [
        ["Inputs", fmt_int(len(inputs))], ["Target", "1" if target else "0"],
        ["Excluded", fmt_int(len(excluded))],
        ["Prepared fields", fmt_int(len([k for k in out_cols
                                         if k != (target or "") + s["suffix"]]))]]))
    section.table(Table("Field Actions", ["Field", "Measurement (prepared)", "Actions"],
                        actions))
    if excluded:
        section.table(Table("Excluded Fields", ["Field", "Reason"],
                            [[a, b] for a, b in excluded]))
    if save:
        created = []
        for name, (vals, kind, label) in out_cols.items():
            nm = name
            if ds.has_name(nm):
                ds.delete_variables([ds.index_of(nm)])
            var = ds.add_variable(nm, values=vals)
            var.label = label
            var.measure = kind
            var.decimals = 3 if kind == "scale" else 0
            created.append(nm)
        section.note("", "Prepared fields: " + ", ".join(created))
    section.result = {"fields": out_cols, "excluded": excluded}  # type: ignore[attr-defined]
    return section


def selftest() -> None:
    r = np.random.default_rng(0)
    x = r.normal(size=200)
    x[0] = 25
    m, sd = robust_mean_sd(x)
    assert abs(m) < 0.3 and sd < 1.5
    lam, _, _ = boxcox_grid(np.exp(r.normal(size=300)))
    assert abs(lam) <= 0.5

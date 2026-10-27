"""Automatic pooled output for multiply imputed data (SPSS *Multiple Imputation* pooling).

When the active dataset is a stacked imputation file (variable ``Imputation_`` with at least
two imputations), SPSS runs a supported procedure on the original data and on every
imputation and adds a **Pooled** row computed with Rubin's (1987) rules to the tables that
support pooling, with *Fraction Missing Info.*, *Relative Increase Variance* and *Relative
Efficiency*. This module does the same:

* model procedures (Linear Regression, Binary/Multinomial Logistic, Ordinal, GENLIN, Cox,
  Linear Mixed Models, GLM Univariate) pool the parameter estimates their result exposes as
  :class:`~rsstatchem.core.stats.resample.ModelInfo` (Barnard-Rubin df for t-based models,
  Rubin's large-sample df for Wald/z-based ones);
* Descriptives, the t tests and Bivariate Correlations use the pooled tables of
  :mod:`rsstatchem.core.stats.imputation` (WP4);
* Frequencies, Means, One-Way ANOVA descriptives, Partial Correlations and Crosstabs counts
  are pooled here.

The per-imputation output follows the pooled tables, labelled "Imputation Number = k", as
SPSS's split-file output does.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_df, fmt_int, fmt_p, numeric, pct, value_text
from rsstatchem.core.stats.imputation import (IMPUTATION_VAR, imputation_variable,
                                              imputation_views, is_imputed, rubin)

ORIGINAL = "Original data"
POOL_COLS = ["Fraction Missing Info.", "Relative Increase Variance", "Relative Efficiency"]


def auto_views(ds: Dataset) -> list[tuple[str, Dataset]] | None:
    """(label, view) for the original data and each imputation, or None when the dataset is
    not a multiply imputed file with at least two imputations."""
    try:
        if not is_imputed(ds):
            return None
        views = imputation_views(ds, include_original=True)
    except (KeyError, ValueError, TypeError):
        return None
    if sum(1 for lb, _ in views if lb != ORIGINAL) < 2:
        return None
    return views


def _info(pl, i: int) -> list[str]:
    return [fmt(pl.fmi[i], 3), fmt(pl.riv[i], 3), fmt(pl.re[i], 3)]


def _dfs(df) -> str:
    return fmt_df(float(df)) if np.isfinite(df) else ""


def models_of(result) -> list:
    models = getattr(result, "models", None)
    if models is None and hasattr(result, "section"):
        models = getattr(result.section, "models", None)
    return list(models or [])


def section_of(result) -> Section:
    return result.section if hasattr(result, "section") else result


def combine(title: str, pooled_tables: list, per_view: list[tuple[str, object]],
            notes: list[str] = (), log: str | None = None) -> Section:
    """Pooled tables first, then each imputation's own output (split-file style)."""
    first = section_of(per_view[0][1])
    out = Section(title)
    logs = [i for i in first.items if getattr(i, "style", "") == "log"]
    out.items.extend(logs)
    m = sum(1 for lb, _ in per_view if lb != ORIGINAL)
    out.note("Multiple Imputation", f"Split by {IMPUTATION_VAR}: output for the original data "
             f"and {m} imputations; pooled estimates use Rubin's rules." +
             (" " + " ".join(notes) if notes else ""))
    for t in pooled_tables:
        out.table(t)
    for label, res in per_view:
        sec = section_of(res)
        out.note(f"Imputation Number = {label}", "")
        for item in sec.items:
            if getattr(item, "style", "") == "log":
                continue
            if getattr(item, "kind", "") == "table":
                item.title = f"{item.title} — Imputation Number = {label}"
            out.items.append(item)
    return out


# ============================================================================ models
def pooled_model_tables(per_view: list[tuple[str, list]], level: float = 0.95) -> list[Table]:
    """One pooled table per model position (e.g. regression Model 1, Model 2)."""
    tables = []
    n_models = min(len(ms) for _, ms in per_view) if per_view else 0
    for pos in range(n_models):
        rows_by_view = [(lb, ms[pos]) for lb, ms in per_view]
        m0 = rows_by_view[0][1]
        key = [tuple(n) if isinstance(n, (list, tuple)) else (n,) for n in m0.names]
        imps = [m for lb, m in rows_by_view if lb != ORIGINAL]
        common = [k for k in key
                  if all(k in [tuple(n) if isinstance(n, (list, tuple)) else (n,)
                               for n in m.names] for m in imps)]

        def pick(m, k):
            ks = [tuple(n) if isinstance(n, (list, tuple)) else (n,) for n in m.names]
            j = ks.index(k)
            return m.estimate[j], m.se[j] if m.se is not None else np.nan, \
                (m.df[j] if isinstance(m.df, np.ndarray) else m.df)

        stat_label = m0.stat_label or ("t" if m0.stat == "t" else "Wald")
        ci = f"{pct(level)} Confidence Interval"
        heads = ["Imputation Number"] + (["Model"] if m0.group else []) + list(m0.headers)
        cols = heads + [m0.est_label, "Std. Error", stat_label, "df", "Sig.",
                        (ci, "Lower Bound"), (ci, "Upper Bound")]
        if m0.exp:
            cols.append("Exp(B)")
        cols += POOL_COLS
        rows = []
        for lb, m in rows_by_view:
            for i, k in enumerate(common):
                b, se, df = pick(m, k)
                rows.append(_row(lb if i == 0 else "", m0, k, i, b, se, df, level, m0.stat))
        q = np.array([[pick(m, k)[0] for k in common] for m in imps])
        u = np.array([[pick(m, k)[1] ** 2 for k in common] for m in imps])
        dfc = pick(imps[0], common[0])[2] if common and m0.stat == "t" else None
        if dfc is not None and isinstance(m0.df, np.ndarray):
            dfc = np.array([pick(imps[0], k)[2] for k in common], float)
        pl = rubin(q, u, df_complete=dfc)
        lo, hi = pl.ci(level)
        for i, k in enumerate(common):
            lead = ["Pooled" if i == 0 else ""] + ([m0.group if i == 0 else ""] if m0.group
                                                   else [])
            lab = list(k) + [""] * (len(m0.headers) - len(k))
            stat = pl.t[i] if m0.stat == "t" else pl.t[i] ** 2
            row = lead + lab + [fmt(pl.estimate[i]), fmt(pl.se[i]), fmt(stat),
                                _dfs(pl.df[i]), fmt_p(pl.p[i]), fmt(lo[i]), fmt(hi[i])]
            if m0.exp:
                row.append(fmt(_exp(pl.estimate[i])))
            rows.append(row + _info(pl, i))
        notes = [f"Pooled with Rubin's rules over {pl.m} imputations"
                 + ("; df use the Barnard-Rubin small-sample adjustment." if dfc is not None
                    else "; df use Rubin's large-sample formula.")]
        if m0.stat != "t":
            notes.append(f"For the pooled row, {stat_label} = t² of the pooled estimate and its "
                         "significance uses Rubin's df.")
        tables.append(Table(f"{m0.table} (Pooled)", cols, rows, row_header_cols=len(heads),
                            footnotes=notes))
    return tables


def _exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def _row(label, m0, key, i, b, se, df, level, stat):
    lead = [label] + ([m0.group if i == 0 else ""] if m0.group else [])
    lab = list(key) + [""] * (len(m0.headers) - len(key))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = b / se
    if stat == "t" and df:
        p = 2 * stats.t.sf(abs(z), df)
        q = stats.t.ppf(0.5 + level / 2, df)
        s, dtext = z, fmt_df(float(df))
    else:
        p = stats.chi2.sf(z * z, 1)
        q = stats.norm.ppf(0.5 + level / 2)
        s, dtext = z * z, "1"
    row = lead + lab + [fmt(b), fmt(se), fmt(s), dtext, fmt_p(p), fmt(b - q * se),
                        fmt(b + q * se)]
    if m0.exp:
        row.append(fmt(_exp(b)))
    return row + [""] * 3


def pool_procedure(title: str, per_view: list[tuple[str, object]], level: float = 0.95
                   ) -> Section:
    """Pooled output for a model procedure run on every imputation."""
    infos = [(lb, models_of(r)) for lb, r in per_view]
    notes = []
    if any(not ms for _, ms in infos):
        notes.append("Pooling is not available for this output (no model estimates).")
        tables = []
    else:
        tables = pooled_model_tables(infos, level)
    return combine(title, tables, per_view, notes)


# ============================================================================ simple statistics
def _pool_scalar(est, var):
    return rubin(np.asarray(est, float)[:, None], np.asarray(var, float)[:, None])


def pooled_frequencies(views, names: list[str]) -> list[Table]:
    rows = []
    for name in names:
        imps_e, imps_v, ns = [], [], []
        for lb, v in views:
            if v.var(name).type != "numeric":
                break
            x = numeric(v, name).dropna().to_numpy()
            mean = x.mean() if len(x) else np.nan
            se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan
            rows.append([v.var(name).title if lb == views[0][0] else "", lb,
                         fmt_int(len(x)), fmt(mean, 4), fmt(se, 5), "", "", ""])
            if lb != ORIGINAL:
                imps_e.append(mean)
                imps_v.append(se ** 2)
                ns.append(len(x))
        if len(imps_e) >= 2:
            pl = _pool_scalar(imps_e, imps_v)
            rows.append(["", "Pooled", fmt(np.mean(ns), 0), fmt(pl.estimate[0], 4),
                         fmt(pl.se[0], 5)] + _info(pl, 0))
    out = [Table("Statistics (Pooled)", ["", "Imputation Number", "N", "Mean",
                                         "Std. Error of Mean"] + POOL_COLS, rows,
                 row_header_cols=2, footnotes=["Pooled Mean and Std. Error by Rubin's rules; "
                                               "pooled N is the average over imputations."])]
    for name in names:
        out.append(_pooled_counts(views, [name], f"{views[0][1].var(name).title} (Pooled)"))
    return out


def _pooled_counts(views, names: list[str], title: str) -> Table:
    var = views[0][1].var(names[0])
    imps = [(lb, v) for lb, v in views if lb != ORIGINAL]
    counts = [v.analysis_series(names[0]).value_counts() for _, v in imps]
    cats = sorted(set().union(*[c.index for c in counts]),
                  key=lambda v: (isinstance(v, str), v))
    mean_counts = np.array([np.mean([c.get(k, 0) for c in counts]) for k in cats])
    total = mean_counts.sum()
    rows = [[value_text(var, k), fmt(mean_counts[i], 1), fmt(100 * mean_counts[i] / total, 1)]
            for i, k in enumerate(cats)]
    rows.append(["Total", fmt(total, 1), "100.0"])
    return Table(title, ["", "Frequency", "Percent"], rows,
                 footnotes=["Pooled frequencies are the averages over the imputations."])


def pooled_means(views, dependents: list[str], layers: list[list[str]]) -> list[Table]:
    layers = [lay for lay in layers if lay]
    combos = list(itertools.product(*layers)) if layers else [()]
    tables = []
    for combo in combos:
        rows = []
        for dep in dependents:
            cell_keys = None
            per = []
            for lb, v in views:
                y = numeric(v, dep)
                if combo:
                    g = pd.concat([v.analysis_series(c) for c in combo], axis=1)
                    frame = pd.concat([y.rename("__y"), g], axis=1).dropna()
                    grouped = {k if isinstance(k, tuple) else (k,): sub["__y"].to_numpy()
                               for k, sub in frame.groupby(list(combo), sort=True)}
                    grouped[("Total",) * len(combo)] = frame["__y"].to_numpy()
                else:
                    grouped = {(): y.dropna().to_numpy()}
                per.append((lb, grouped))
                cell_keys = list(grouped) if cell_keys is None else \
                    [k for k in cell_keys if k in grouped]
            for key in cell_keys:
                label = " * ".join(
                    "Total" if k == "Total" else value_text(views[0][1].var(c), k)
                    for c, k in zip(combo, key)) or "Total"
                est, var_, ns = [], [], []
                for lb, grouped in per:
                    x = grouped[key]
                    mean = x.mean() if len(x) else np.nan
                    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan
                    rows.append([views[0][1].var(dep).title, label, lb, fmt(mean, 4),
                                 fmt_int(len(x)), fmt(x.std(ddof=1), 5) if len(x) > 1 else "",
                                 "", "", ""])
                    if lb != ORIGINAL:
                        est.append(mean)
                        var_.append(se ** 2)
                        ns.append(len(x))
                pl = _pool_scalar(est, var_)
                rows.append(["", "", "Pooled", fmt(pl.estimate[0], 4), fmt(np.mean(ns), 0),
                             ""] + _info(pl, 0))
        heads = " * ".join(views[0][1].var(c).title for c in combo) or "Cell"
        tables.append(Table("Report (Pooled)", ["", heads, "Imputation Number", "Mean", "N",
                                                "Std. Deviation"] + POOL_COLS, rows,
                            row_header_cols=3,
                            footnotes=["Pooled Mean by Rubin's rules (within-imputation "
                                       "variance s²/n); pooled N is the average."]))
    return tables


def pooled_oneway(views, dependents: list[str], factor: str, level: float = 0.95
                  ) -> list[Table]:
    rows = []
    ci = f"{pct(level)} Confidence Interval for Mean"
    for dep in dependents:
        per = []
        for lb, v in views:
            frame = pd.DataFrame({"y": numeric(v, dep), "g": v.analysis_series(factor)}).dropna()
            grouped = {k: sub["y"].to_numpy() for k, sub in frame.groupby("g", sort=True)}
            grouped["Total"] = frame["y"].to_numpy()
            per.append((lb, grouped))
        keys = [k for k in per[0][1] if all(k in g for _, g in per)]
        keys = [k for k in keys if k != "Total"] + ["Total"]
        fvar = views[0][1].var(factor)
        for key in keys:
            label = "Total" if key == "Total" else value_text(fvar, key)
            est, var_, ns = [], [], []
            for lb, grouped in per:
                x = grouped[key]
                n = len(x)
                mean = x.mean()
                se = x.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
                q = stats.t.ppf(0.5 + level / 2, n - 1) if n > 1 else np.nan
                rows.append([views[0][1].var(dep).title, label, lb, fmt_int(n), fmt(mean, 4),
                             fmt(x.std(ddof=1), 5) if n > 1 else "", fmt(se, 5),
                             fmt(mean - q * se, 4), fmt(mean + q * se, 4), "", "", ""])
                if lb != ORIGINAL:
                    est.append(mean)
                    var_.append(se ** 2)
                    ns.append(n)
            pl = _pool_scalar(est, var_)
            lo, hi = pl.ci(level)
            rows.append(["", "", "Pooled", fmt(np.mean(ns), 0), fmt(pl.estimate[0], 4), "",
                         fmt(pl.se[0], 5), fmt(lo[0], 4), fmt(hi[0], 4)] + _info(pl, 0))
    return [Table("Descriptives (Pooled)", ["", "", "Imputation Number", "N", "Mean",
                                            "Std. Deviation", "Std. Error", (ci, "Lower Bound"),
                                            (ci, "Upper Bound")] + POOL_COLS, rows,
                  row_header_cols=3)]


def pooled_partial(views, names: list[str], controls: list[str]) -> list[Table]:
    rows = []
    q = len(controls)
    for a, b in itertools.combinations(names, 2):
        zs, vs = [], []
        for lb, v in views:
            frame = pd.concat({n: numeric(v, n) for n in dict.fromkeys([a, b] + controls)},
                              axis=1).dropna()
            R = np.corrcoef(frame.to_numpy(float).T)
            P = np.linalg.inv(R)
            r = -P[0, 1] / math.sqrt(P[0, 0] * P[1, 1])
            df = len(frame) - 2 - q
            t = r * math.sqrt(df / max(1 - r * r, 1e-300))
            rows.append([f"{a} - {b}", lb, fmt(r), fmt_p(2 * stats.t.sf(abs(t), df)),
                         fmt_int(df), "", "", ""])
            if lb != ORIGINAL:
                zs.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
                vs.append(1 / (len(frame) - 3 - q))
        pl = _pool_scalar(zs, vs)
        zstat = pl.estimate[0] / pl.se[0]
        p = 2 * stats.norm.sf(abs(zstat)) if not np.isfinite(pl.df[0]) else pl.p[0]
        rows.append(["", "Pooled", fmt(np.tanh(pl.estimate[0])), fmt_p(p), "", ] + _info(pl, 0))
    return [Table("Correlations (Pooled)", ["Pair", "Imputation Number", "Correlation",
                                            "Significance (2-tailed)", "df"] + POOL_COLS, rows,
                  row_header_cols=2,
                  footnotes=[f"Control variables: {', '.join(controls)}.",
                             "Pooled partial correlations are back-transformed means of "
                             "Fisher's z (variance 1/(n-3-q)); significance by Rubin's rules."])]


def pooled_crosstab(views, row: str, col: str) -> Table:
    imps = [v for lb, v in views if lb != ORIGINAL]
    tabs = [pd.crosstab(v.analysis_series(row), v.analysis_series(col)) for v in imps]
    rcats = sorted(set().union(*[t.index for t in tabs]), key=lambda v: (isinstance(v, str), v))
    ccats = sorted(set().union(*[t.columns for t in tabs]),
                   key=lambda v: (isinstance(v, str), v))
    M = np.mean([t.reindex(index=rcats, columns=ccats, fill_value=0).to_numpy(float)
                 for t in tabs], axis=0)
    rv, cv = views[0][1].var(row), views[0][1].var(col)
    rows = [[rv.title if i == 0 else "", value_text(rv, r), "Count"] +
            [fmt(M[i, j], 1) for j in range(len(ccats))] + [fmt(M[i].sum(), 1)]
            for i, r in enumerate(rcats)]
    rows.append(["Total", "", "Count"] + [fmt(x, 1) for x in M.sum(axis=0)] +
                [fmt(M.sum(), 1)])
    return Table(f"{rv.title} * {cv.title} Crosstabulation (Pooled)",
                 ["", "", ""] + [(cv.title, value_text(cv, c)) for c in ccats] + ["Total"],
                 rows, row_header_cols=3,
                 footnotes=["Pooled counts are the averages over the imputations."])


def selftest() -> None:
    rng = np.random.default_rng(5)
    frames = []
    for k in range(4):
        x = rng.normal(size=30)
        frames.append(pd.DataFrame({IMPUTATION_VAR: float(k), "x": x,
                                    "y": 1 + 2 * x + rng.normal(size=30),
                                    "grp": np.tile([1.0, 2.0, 3.0], 10)}))
    ds = Dataset(pd.concat(frames, ignore_index=True))
    views = auto_views(ds)
    assert views is not None and len(views) == 4 and imputation_variable(ds)
    assert pooled_frequencies(views, ["x"])[0].rows[-1][1] == "Pooled"
    assert pooled_oneway(views, ["y"], "grp")[0].rows[-1][2] == "Pooled"

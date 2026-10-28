"""Analyze > Survival: Life Tables, Kaplan-Meier, Cox Regression and Cox with time-dependent
covariates, mirroring SPSS SURVIVAL, KM and COXREG.

* Kaplan-Meier: product-limit estimates with Greenwood standard errors, the per-case survival
  table, restricted means, medians and quartiles with Collett's standard errors (as SPSS), the
  Log Rank (Mantel-Cox), Breslow (generalized Wilcoxon) and Tarone-Ware tests (pooled over strata,
  per stratum, pairwise, linear trend) and survival / one minus survival / hazard / log survival
  plots. Results agree with ``statsmodels`` SurvfuncRight/survdiff and lifelines.
* Life Tables: actuarial estimates (exposed = entering - withdrawn/2), density and hazard with
  their standard errors, interpolated median, Wilcoxon (Gehan) comparison via the Lee-Desu
  statistic.
* Cox regression: own Newton-Raphson partial likelihood with Breslow ties (the SPSS method),
  strata, counting-process (start, stop] data, categorical covariates with SPSS contrasts, Enter /
  Forward (Conditional, LR, Wald) / Backward (Conditional, LR, Wald) selection with score tests
  for entry, omnibus tests, baseline survival, plots and SPSS's saved variables. Estimates agree
  with ``statsmodels.PHReg(ties="breslow")``. Time-dependent covariates are expressions of
  ``T_`` evaluated on episodes split at every event time.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.dataset import NUMERIC, Dataset, Variable
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, value_text
from rsstatchem.core.stats.wp4_util import (COLORS, CONTRASTS, Coding, Result, describe_event,
                                            event_mask, figure, make_coding, png, raw_series,
                                            sort_key)

TESTS = {"logrank": "Log Rank (Mantel-Cox)", "breslow": "Breslow (Generalized Wilcoxon)",
         "tarone": "Tarone-Ware"}
COX_METHODS = {"enter": "Enter", "fstep_cond": "Forward: Conditional", "fstep_lr": "Forward: LR",
               "fstep_wald": "Forward: Wald", "bstep_cond": "Backward: Conditional",
               "bstep_lr": "Backward: LR", "bstep_wald": "Backward: Wald"}
_REF_CONTRASTS = ("indicator", "simple", "deviation")
KM_PLOTS = {"survival": "Survival", "one_minus": "One minus survival", "hazard": "Hazard",
            "log_survival": "Log survival"}


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


def _sig(p) -> str:
    return fmt_p(p) if p is not None and p == p else ""


# =========================================================================== data preparation
@dataclass
class SurvData:
    frame: pd.DataFrame            # t, d and optional f (factor), s (strata) columns
    n_total: int
    n_missing: int
    n_negative: int


def _prepare(ds: Dataset, time: str, status: str, event, factor: str | None = None,
             strata: str | None = None, extra: list[str] | None = None) -> SurvData:
    t = raw_series(ds, time)
    d = event_mask(raw_series(ds, status), event)
    cols = {"t": t.astype(float), "d": d}
    if factor:
        cols["f"] = raw_series(ds, factor)
    if strata:
        cols["s"] = raw_series(ds, strata)
    for name in extra or []:
        cols[name] = raw_series(ds, name)
    frame = pd.DataFrame(cols)
    n_total = len(frame)
    complete = frame.notna().all(axis=1)
    negative = complete & (frame["t"] < 0)
    frame = frame[complete & ~negative]
    return SurvData(frame, n_total, int((~complete).sum()), int(negative.sum()))


def _level_label(ds: Dataset, name: str | None, value) -> str:
    return value_text(ds.var(name), value) if name else ""


# =========================================================================== Kaplan-Meier core
@dataclass
class KMCurve:
    times: np.ndarray        # distinct event times
    n_risk: np.ndarray
    n_event: np.ndarray
    surv: np.ndarray
    se: np.ndarray
    t: np.ndarray            # all times, sorted (events before censored at ties)
    d: np.ndarray
    n: int

    @property
    def events(self) -> int:
        return int(self.d.sum())

    @property
    def last_time(self) -> float:
        return float(self.t[-1]) if self.n else float("nan")

    def at(self, x) -> np.ndarray:
        """S(x), right-continuous step function."""
        idx = np.searchsorted(self.times, np.asarray(x, float), side="right") - 1
        return np.where(idx >= 0, self.surv[np.clip(idx, 0, None)] if len(self.surv) else 1.0,
                        1.0)

    def se_at(self, x) -> np.ndarray:
        idx = np.searchsorted(self.times, np.asarray(x, float), side="right") - 1
        return np.where(idx >= 0, self.se[np.clip(idx, 0, None)] if len(self.se) else 0.0, 0.0)


def km_fit(t, d) -> KMCurve:
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    order = np.lexsort((-d, t))
    t, d = t[order], d[order]
    times = np.unique(t[d == 1])
    n_risk = len(t) - np.searchsorted(t, times, side="left")
    te = t[d == 1]
    n_event = np.searchsorted(te, times, side="right") - np.searchsorted(te, times, side="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(1.0 - n_event / n_risk)
        terms = np.where(n_risk > n_event, n_event / (n_risk * (n_risk - n_event)), 0.0)
        se = surv * np.sqrt(np.cumsum(terms))
    return KMCurve(times, n_risk.astype(float), n_event.astype(float), surv, se, t, d, len(t))


def km_mean(c: KMCurve) -> tuple[float, float, bool]:
    """Mean survival (area under S up to the largest time) and its SE; the flag tells whether
    the largest time is censored (estimate restricted)."""
    if c.n == 0:
        return float("nan"), float("nan"), False
    tmax = c.last_time
    knots = np.concatenate([[0.0], c.times[c.times <= tmax], [tmax]])
    heights = np.concatenate([[1.0], c.surv[c.times <= tmax]])
    widths = np.diff(knots)
    mean = float(np.sum(heights * widths))
    # A_j: area under S beyond the j-th event time
    seg = heights[1:] * widths[1:]
    area_after = np.cumsum(seg[::-1])[::-1]
    nr, ne = c.n_risk, c.n_event
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(nr > ne, area_after ** 2 * ne / (nr * (nr - ne)), 0.0)
    restricted = bool(c.d[-1] == 0) if c.n else False
    return mean, float(math.sqrt(np.sum(terms))), restricted


def km_percentile(c: KMCurve, p: float, eps: float = 0.05) -> tuple[float, float]:
    """Time t_p with S(t_p) <= 1 - p (SPSS percentiles are on the survival scale: the 25%
    percentile is where 25% survive), and Collett's SE = SE(S(t_p)) / f(t_p)."""
    target = 1.0 - p
    hit = np.nonzero(c.surv <= target + 1e-12)[0]
    if not len(hit):
        return float("nan"), float("nan")
    j = hit[0]
    tp = float(c.times[j])
    up = np.nonzero(c.surv >= target + eps)[0]
    lo = np.nonzero(c.surv <= target - eps)[0]
    u = float(c.times[up[-1]]) if len(up) else 0.0
    su = float(c.surv[up[-1]]) if len(up) else 1.0
    if not len(lo):
        return tp, float("nan")
    l_ = float(c.times[lo[0]])
    sl = float(c.surv[lo[0]])
    if l_ <= u:
        return tp, float("nan")
    dens = (su - sl) / (l_ - u)
    return tp, float(c.se[j] / dens) if dens > 0 else float("nan")


def compare_groups(t, d, g, strata=None, weight: str = "logrank", levels=None,
                   trend: bool = False) -> tuple[float, int, float]:
    """k-sample test of equal survival (Log Rank / Breslow / Tarone-Ware), pooled over strata.
    With ``trend`` the 1-df test for linear trend with equally spaced scores."""
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    g = np.asarray(g, dtype=object)
    s = np.zeros(len(t)) if strata is None else np.asarray(strata, dtype=object)
    levels = list(levels) if levels is not None else sorted(set(g), key=sort_key)
    k = len(levels)
    if k < 2:
        return float("nan"), 0, float("nan")
    u = np.zeros(k)
    v = np.zeros((k, k))
    for stratum in sorted(set(s.tolist()), key=sort_key):
        m = s == stratum
        ts, dd, gs = t[m], d[m], g[m]
        et = np.unique(ts[dd == 1])
        if not len(et):
            continue
        nmat = np.zeros((len(et), k))
        dmat = np.zeros((len(et), k))
        for i, lv in enumerate(levels):
            tg = np.sort(ts[gs == lv])
            nmat[:, i] = len(tg) - np.searchsorted(tg, et, side="left")
            te = np.sort(ts[(gs == lv) & (dd == 1)])
            dmat[:, i] = np.searchsorted(te, et, "right") - np.searchsorted(te, et, "left")
        n = nmat.sum(1)
        dt = dmat.sum(1)
        w = {"logrank": np.ones_like(n), "breslow": n, "tarone": np.sqrt(n)}[weight]
        u += (w[:, None] * (dmat - nmat * (dt / n)[:, None])).sum(0)
        c = w ** 2 * dt * (n - dt) / np.maximum(n - 1, 1) / n
        v += np.diag((c[:, None] * nmat).sum(0)) - (nmat.T * (c / n)) @ nmat
    if trend:
        sc = np.arange(1, k + 1, dtype=float)
        den = sc @ v @ sc
        if den <= 0:
            return float("nan"), 1, float("nan")
        chi = float((sc @ u) ** 2 / den)
        return chi, 1, float(stats.chi2.sf(chi, 1))
    keep = np.diag(v) > 1e-12
    idx = np.nonzero(keep)[0][:-1] if keep.sum() == k else np.nonzero(keep)[0][:-1]
    if len(idx) == 0:
        return float("nan"), 0, float("nan")
    sub = v[np.ix_(idx, idx)]
    chi = float(u[idx] @ np.linalg.pinv(sub) @ u[idx])
    df = int(np.linalg.matrix_rank(sub))
    return chi, df, float(stats.chi2.sf(chi, df)) if df else float("nan")


# =========================================================================== Kaplan-Meier
def _step_xy(times, values, start_value, tmax):
    x = np.concatenate([[0.0], np.repeat(times, 2), [tmax]])
    y = np.concatenate([[start_value, start_value], np.repeat(values, 2)])
    y = y[:len(x)]
    return x, y


def _km_plot(curves: list[tuple[str, KMCurve]], kind: str, title: str, xlabel: str) -> bytes:
    fig = figure(6.2, 4.4)
    ax = fig.add_subplot(111)
    for i, (label, c) in enumerate(curves):
        color = COLORS[i % len(COLORS)]
        tmax = max(c.last_time, c.times[-1] if len(c.times) else 0.0)
        s = c.surv
        with np.errstate(divide="ignore"):
            vals = {"survival": s, "one_minus": 1 - s, "hazard": -np.log(s),
                    "log_survival": np.log(s)}[kind]
        start = {"survival": 1.0, "one_minus": 0.0, "hazard": 0.0, "log_survival": 0.0}[kind]
        finite = np.isfinite(vals)
        x, y = _step_xy(c.times[finite], vals[finite], start, tmax)
        ax.plot(x, y, color=color, lw=1.4, label=label or None)
        cens = c.t[c.d == 0]
        if len(cens):
            sc = c.at(cens)
            with np.errstate(divide="ignore"):
                yc = {"survival": sc, "one_minus": 1 - sc, "hazard": -np.log(sc),
                      "log_survival": np.log(sc)}[kind]
            ok = np.isfinite(yc)
            ax.plot(cens[ok], yc[ok], "+", color=color, ms=7, mew=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel({"survival": "Cum Survival", "one_minus": "Cum Survival (1 - S)",
                   "hazard": "Cum Hazard", "log_survival": "Log Survival"}[kind])
    ax.set_title(title)
    if any(label for label, _ in curves):
        ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return png(fig)


def kaplan_meier(ds: Dataset, time: str, status: str, event, factor: str | None = None,
                 strata: str | None = None, label_var: str | None = None,
                 tests: tuple = (), compare: str = "pooled", trend: bool = False,
                 survival_table: bool = True, mean_median: bool = True,
                 quartiles: bool = False, plots: tuple = (), save: tuple = (),
                 level: float = 0.95) -> Result:
    """SPSS KM. ``compare``: pooled | stratum (for each stratum) | pairwise (pairwise over
    strata) | pairwise_stratum. ``save``: survival, se, hazard, cum_events."""
    sd = _prepare(ds, time, status, event, factor, strata)
    fr = sd.frame
    sec = Section("Kaplan-Meier")
    syntax = f"KM {time}{' BY ' + factor if factor else ''}" \
             f"{' STRATA=' + strata if strata else ''}\n  /STATUS={status}{describe_event(event)}"
    if tests:
        syntax += "\n  /TEST " + " ".join({"logrank": "LOGRANK", "breslow": "BRESLOW",
                                          "tarone": "TARONE"}[x] for x in tests)
        syntax += f"\n  /COMPARE={'OVERALL' if 'pairwise' not in compare else 'PAIRWISE'} " \
                  f"{'POOLED' if 'stratum' not in compare else 'STRATA'}"
    sec.log(syntax + ".")
    if fr.empty:
        sec.note("Warning", "There are no valid cases for this analysis.", "warning")
        return Result(sec)
    flevels = sorted(fr["f"].unique(), key=sort_key) if factor else [None]
    slevels = sorted(fr["s"].unique(), key=sort_key) if strata else [None]
    tvar = ds.var(time)
    status_var = ds.var(status)

    def subset(fv, sv):
        m = np.ones(len(fr), bool)
        if fv is not None:
            m &= (fr["f"] == fv).to_numpy()
        if sv is not None:
            m &= (fr["s"] == sv).to_numpy()
        return fr[m]

    groups = [(fv, sv) for sv in slevels for fv in flevels]
    curves = {key: km_fit(subset(*key)["t"], subset(*key)["d"]) for key in groups}

    def glabel(fv, sv):
        parts = []
        if strata:
            parts.append(_level_label(ds, strata, sv))
        if factor:
            parts.append(_level_label(ds, factor, fv))
        return " / ".join(parts) if parts else ""

    # Case Processing Summary
    hdr = [c for c in (ds.var(strata).title if strata else None,
                       ds.var(factor).title if factor else None) if c]
    rows = []
    for key in groups:
        c = curves[key]
        if c.n == 0:
            continue
        lead = []
        if strata:
            lead.append(_level_label(ds, strata, key[1]))
        if factor:
            lead.append(_level_label(ds, factor, key[0]))
        cens = c.n - c.events
        rows.append(lead + [fmt_int(c.n), fmt_int(c.events), fmt_int(cens),
                            _f(100 * cens / c.n, 1) + "%"])
    if factor or strata:
        n_all = len(fr)
        ev = int(fr["d"].sum())
        rows.append(["Overall"] + [""] * (len(hdr) - 1) +
                    [fmt_int(n_all), fmt_int(ev), fmt_int(n_all - ev),
                     _f(100 * (n_all - ev) / n_all, 1) + "%"])
    else:
        hdr = [""]
        for r in rows:
            r.insert(0, "")
    sec.table(Table("Case Processing Summary",
                    hdr + ["Total N", "N of Events", ("Censored", "N"), ("Censored", "Percent")],
                    rows, row_header_cols=len(hdr)))

    # Survival Table
    level_pct = f"{level * 100:g}%"
    if survival_table:
        trows = []
        for key in groups:
            c = curves[key]
            if c.n == 0:
                continue
            gl = glabel(*key)
            cum_events = np.cumsum(c.d)
            last_of_tie = np.ones(c.n, bool)
            for i in range(c.n - 1):
                if c.d[i] == 1 and c.d[i + 1] == 1 and c.t[i + 1] == c.t[i]:
                    last_of_tie[i] = False
            sub = subset(*key)
            if label_var:
                lab_series = ds.analysis_series(label_var).loc[sub.index]
                order = np.lexsort((-sub["d"].to_numpy(), sub["t"].to_numpy()))
                labels = lab_series.to_numpy(dtype=object)[order]
            else:
                labels = None
            for i in range(c.n):
                est = se = ""
                if c.d[i] == 1 and last_of_tie[i]:
                    est, se = _f(float(c.at(c.t[i])), 3), _f(float(c.se_at(c.t[i])), 3)
                code = 1.0 if c.d[i] else 0.0
                status_text = "Event" if code else "Censored"
                row = [gl, str(i + 1)]
                if labels is not None:
                    row.append(str(labels[i]))
                row += [fmt(c.t[i], max(tvar.decimals, 0)), status_text, est, se,
                        fmt_int(cum_events[i]), fmt_int(c.n - i - 1)]
                trows.append(row)
        cols = ["", ""] + ([ds.var(label_var).title] if label_var else []) + [
            "Time", "Status",
            ("Cumulative Proportion Surviving at the Time", "Estimate"),
            ("Cumulative Proportion Surviving at the Time", "Std. Error"),
            "N of Cumulative Events", "N of Remaining Cases"]
        if not (factor or strata):
            for r in trows:
                r.pop(0)
            cols = cols[1:]
        sec.table(Table("Survival Table", cols, trows,
                        row_header_cols=2 if (factor or strata) else 1))

    # Means and Medians
    z = stats.norm.ppf(0.5 + level / 2)
    if mean_median:
        mrows = []
        restricted_any = False
        for key in groups + ([("__overall__", None)] if (factor or strata) else []):
            c = curves[key] if key in curves else km_fit(fr["t"], fr["d"])
            if c.n == 0:
                continue
            mean, mse, restricted = km_mean(c)
            restricted_any |= restricted
            med, medse = km_percentile(c, 0.5)
            label = "Overall" if key[0] == "__overall__" else glabel(*key)
            mrows.append([label, _f(mean, 3) + ("ᵃ" if restricted else ""), _f(mse, 3),
                          _f(mean - z * mse, 3), _f(mean + z * mse, 3),
                          _f(med, 3), _f(medse, 3),
                          _f(med - z * medse, 3) if medse == medse else "",
                          _f(med + z * medse, 3) if medse == medse else ""])
        ci = f"{level_pct} Confidence Interval"
        cols = [""] + [(g, s) for g in ("Meanᵃ", "Median")
                       for s in ("Estimate", "Std. Error", f"{ci} Lower Bound",
                                 f"{ci} Upper Bound")]
        notes = ["a. Estimation is limited to the largest survival time if it is censored."]
        sec.table(Table("Means and Medians for Survival Time", cols, mrows, footnotes=notes))
    if quartiles:
        qrows = []
        for key in groups:
            c = curves[key]
            if c.n == 0:
                continue
            row = [glabel(*key)]
            for p in (0.25, 0.5, 0.75):
                est, se = km_percentile(c, 1 - p)
                row += [_f(est, 3), _f(se, 3)]
            qrows.append(row)
        cols = [""] + [(f"{p:.1f}%", s) for p in (25.0, 50.0, 75.0)
                       for s in ("Estimate", "Std. Error")]
        sec.table(Table("Percentiles", cols, qrows))

    # Tests
    if tests and factor and len(flevels) > 1:
        t_all, d_all = fr["t"].to_numpy(), fr["d"].to_numpy()
        f_all = fr["f"].to_numpy(dtype=object)
        s_all = fr["s"].to_numpy(dtype=object) if strata else None
        fname = ds.var(factor).title
        foot = [f"Test of equality of survival distributions for the different levels of "
                f"{fname}."]
        if strata and "stratum" not in compare:
            foot = [f"Adjusted for {ds.var(strata).title}."] + foot
        if trend:
            foot = [f"Test of equality of survival distributions for the different levels of "
                    f"{fname} (linear trend)."]
        if compare in ("pooled", "stratum"):
            rows = []
            for sv in (slevels if compare == "stratum" and strata else [None]):
                m = np.ones(len(t_all), bool) if sv is None else (s_all == sv)
                for test in tests:
                    chi, df, p = compare_groups(t_all[m], d_all[m], f_all[m],
                                                None if (sv is not None or s_all is None)
                                                else s_all, test, trend=trend)
                    lead = [_level_label(ds, strata, sv)] if compare == "stratum" and strata \
                        else []
                    rows.append(lead + [TESTS[test], _f(chi, 3), fmt_int(df), _sig(p)])
            cols = ([ds.var(strata).title] if compare == "stratum" and strata else []) + \
                ["", "Chi-Square", "df", "Sig."]
            sec.table(Table("Overall Comparisons", cols, rows,
                            row_header_cols=2 if compare == "stratum" and strata else 1,
                            footnotes=foot))
        else:
            labels = [_level_label(ds, factor, v) for v in flevels]
            rows = []
            for sv in (slevels if compare == "pairwise_stratum" and strata else [None]):
                m0 = np.ones(len(t_all), bool) if sv is None else (s_all == sv)
                for test in tests:
                    for i, a in enumerate(flevels):
                        row = [_level_label(ds, strata, sv)] if sv is not None else []
                        row += [TESTS[test] if i == 0 else "", labels[i]]
                        for b in flevels:
                            if a == b:
                                row += ["", ""]
                                continue
                            m = m0 & np.isin(f_all, [a, b])
                            chi, _df, p = compare_groups(
                                t_all[m], d_all[m], f_all[m],
                                None if (sv is not None or s_all is None) else s_all[m], test,
                                levels=[a, b])
                            row += [_f(chi, 3), _sig(p)]
                        rows.append(row)
            cols = (([ds.var(strata).title] if compare == "pairwise_stratum" and strata else [])
                    + ["", fname] + [(lb, s) for lb in labels for s in ("Chi-Square", "Sig.")])
            sec.table(Table("Pairwise Comparisons", cols, rows,
                            row_header_cols=3 if compare == "pairwise_stratum" and strata else 2,
                            footnotes=foot))

    # Plots
    for kind in plots:
        if strata and factor:
            for sv in slevels:
                cl = [(_level_label(ds, factor, fv), curves[(fv, sv)]) for fv in flevels
                      if curves[(fv, sv)].n]
                title = f"{KM_PLOTS[kind]} Functions: {ds.var(strata).title} = " \
                        f"{_level_label(ds, strata, sv)}"
                sec.image(title, _km_plot(cl, kind, title, tvar.title))
        else:
            cl = [(glabel(*key), curves[key]) for key in groups if curves[key].n]
            title = f"{KM_PLOTS[kind]} Functions"
            sec.image(title, _km_plot(cl, kind, title, tvar.title))

    saved = {}
    if save:
        index = ds.df.index
        out = {k: pd.Series(np.nan, index=index) for k in ("survival", "se", "hazard",
                                                            "cum_events")}
        for key in groups:
            sub = subset(*key)
            c = curves[key]
            if sub.empty:
                continue
            ts = sub["t"].to_numpy()
            s = c.at(ts)
            out["survival"].loc[sub.index] = s
            out["se"].loc[sub.index] = c.se_at(ts)
            with np.errstate(divide="ignore"):
                out["hazard"].loc[sub.index] = -np.log(s)
            order = np.lexsort((-sub["d"].to_numpy(), ts))
            cum = np.empty(len(ts))
            cum[order] = np.cumsum(sub["d"].to_numpy()[order])
            out["cum_events"].loc[sub.index] = cum
        names = {"survival": ("SUR", "Survival Function"), "se": ("SE", "Standard Error of "
                 "Survival Function"), "hazard": ("HAZ", "Cumulative Hazard Function"),
                 "cum_events": ("CUM", "Cumulative Number of Events")}
        for key in save:
            base, label = names[key]
            saved[base] = (label, out[key].replace([np.inf, -np.inf], np.nan), 5)
    _ = status_var
    return Result(sec, saved)


# =========================================================================== Life Tables
def actuarial(t, d, width: float, end: float) -> pd.DataFrame:
    """Life table rows for intervals [0, w), [w, 2w), ..., [end, inf)."""
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    starts = np.arange(0.0, end + width * 1e-9, width)
    if not len(starts):
        starts = np.array([0.0])
    bins = np.searchsorted(starts, t, side="right") - 1
    k = len(starts)
    deaths = np.bincount(bins[d == 1], minlength=k)[:k].astype(float)
    withdrawn = np.bincount(bins[d == 0], minlength=k)[:k].astype(float)
    total = deaths + withdrawn
    entering = len(t) - np.concatenate([[0.0], np.cumsum(total)[:-1]])
    exposed = entering - withdrawn / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(exposed > 0, deaths / exposed, 0.0)
        p = 1 - q
        cum = np.cumprod(p)
        prev = np.concatenate([[1.0], cum[:-1]])
        term = np.where((exposed > 0) & (p > 0), q / (exposed * p), 0.0)
        se_cum = cum * np.sqrt(np.cumsum(term))
        h = width
        dens = (prev - cum) / h
        prior = np.concatenate([[0.0], np.cumsum(term)[:-1]])
        inner = prior + np.where(q > 0, p / (exposed * q), 0.0)
        se_dens = np.where(q > 0, prev * q / h * np.sqrt(inner), 0.0)
        hazard = np.where(exposed > 0, 2 * deaths / (h * (2 * exposed - deaths)), 0.0)
        se_haz = np.where(deaths > 0, hazard * np.sqrt(
            np.clip(1 - (h * hazard / 2) ** 2, 0, None) / (exposed * q)), 0.0)
    df = pd.DataFrame({"start": starts, "entering": entering, "withdrawn": withdrawn,
                       "exposed": exposed, "events": deaths, "q": q, "p": p, "cum": cum,
                       "se_cum": se_cum, "density": dens, "se_density": se_dens,
                       "hazard": hazard, "se_hazard": se_haz})
    # the last interval is open: no width, so density and hazard are not estimable
    df.loc[df.index[-1], ["density", "se_density", "hazard", "se_hazard"]] = np.nan
    return df[df["entering"] > 0].reset_index(drop=True) if (df["entering"] > 0).any() else df


def life_table_median(lt: pd.DataFrame, width: float) -> tuple[float, bool]:
    prev = 1.0
    for _, r in lt.iterrows():
        if r["cum"] <= 0.5:
            if math.isnan(r["density"]):  # open last interval
                return float(r["start"]), True
            if prev == r["cum"]:
                return float(r["start"]), False
            return float(r["start"] + width * (prev - 0.5) / (prev - r["cum"])), False
        prev = r["cum"]
    return float(lt["start"].iloc[-1]), True


def lee_desu(t, d, g) -> tuple[float, int, float]:
    """Wilcoxon (Gehan) comparison of k groups with the Lee-Desu statistic (SPSS SURVIVAL)."""
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    g = np.asarray(g, dtype=object)
    n = len(t)
    ev = np.sort(t[d == 1])
    allt = np.sort(t)
    cens = np.sort(t[d == 0])
    less_events = np.searchsorted(ev, t, "left")                        # events strictly before
    tie_events = np.searchsorted(ev, t, "right") - less_events          # events at same time
    greater_all = n - np.searchsorted(allt, t, "right")                 # anything strictly after
    tie_cens = np.searchsorted(cens, t, "right") - np.searchsorted(cens, t, "left")
    u = less_events + np.where(d == 0, tie_events, 0) - \
        np.where(d == 1, greater_all + tie_cens, 0)
    levels = sorted(set(g.tolist()), key=sort_key)
    if len(levels) < 2 or np.sum(u ** 2) == 0:
        return float("nan"), 0, float("nan")
    w = sum(u[g == lv].sum() ** 2 / (g == lv).sum() for lv in levels)
    stat = float((n - 1) * w / np.sum(u.astype(float) ** 2))
    df = len(levels) - 1
    return stat, df, float(stats.chi2.sf(stat, df))


def life_tables(ds: Dataset, time: str, status: str, event, width: float, end: float,
                factor: str | None = None, factor_range: tuple | None = None,
                by: str | None = None, show_tables: bool = True, plots: tuple = (),
                compare: bool = False, pairwise: bool = False) -> Section:
    """SPSS SURVIVAL. ``plots`` from survival, log_survival, hazard, density, one_minus."""
    extra = [by] if by else []
    sd = _prepare(ds, time, status, event, factor, None, extra)
    fr = sd.frame
    if factor and factor_range:
        fr = fr[(fr["f"] >= factor_range[0]) & (fr["f"] <= factor_range[1])]
    sec = Section("Life Tables")
    syntax = f"SURVIVAL TABLE={time}{' BY ' + factor if factor else ''}" \
             f"{' BY ' + by if by else ''}\n  /INTERVAL=THRU {end:g} BY {width:g}\n" \
             f"  /STATUS={status}{describe_event(event)}"
    if plots:
        syntax += "\n  /PLOTS (" + " ".join(p.upper() for p in plots) + ")"
    if compare:
        syntax += "\n  /COMPARE=" + ("PAIRWISE" if pairwise else "ALL")
    sec.log(syntax + ".")
    if fr.empty:
        sec.note("Warning", "There are no valid cases for this analysis.", "warning")
        return sec
    tvar = ds.var(time)
    blevels = sorted(fr[by].unique(), key=sort_key) if by else [None]
    flevels = sorted(fr["f"].unique(), key=sort_key) if factor else [None]
    rows, medians = [], []
    curves = {}
    for bv in blevels:
        for fv in flevels:
            m = np.ones(len(fr), bool)
            if bv is not None:
                m &= (fr[by] == bv).to_numpy()
            if fv is not None:
                m &= (fr["f"] == fv).to_numpy()
            sub = fr[m]
            if sub.empty:
                continue
            lt = actuarial(sub["t"], sub["d"], width, end)
            curves[(bv, fv)] = lt
            lead = []
            if by:
                lead.append(_level_label(ds, by, bv))
            if factor:
                lead.append(_level_label(ds, factor, fv))
            for i, r in lt.iterrows():
                rows.append([*(lead if i == 0 else [""] * len(lead)),
                             fmt(r["start"], 3), _f(r["entering"], 3), _f(r["withdrawn"], 3),
                             _f(r["exposed"], 3), _f(r["events"], 3), fmt(r["q"], 2),
                             fmt(r["p"], 2), fmt(r["cum"], 2), fmt(r["se_cum"], 2),
                             fmt(r["density"], 3), fmt(r["se_density"], 3),
                             fmt(r["hazard"], 2), fmt(r["se_hazard"], 2)])
            med, open_ = life_table_median(lt, width)
            medians.append(lead + [fmt(med, 2) + ("+" if open_ else "")])
    lead_cols = [c for c in ((ds.var(by).title if by else None),
                             (ds.var(factor).title if factor else None)) if c]
    if show_tables:
        cols = lead_cols + ["Interval Start Time", "Number Entering Interval",
                            "Number Withdrawing during Interval", "Number Exposed to Risk",
                            "Number of Terminal Events", "Proportion Terminating",
                            "Proportion Surviving",
                            "Cumulative Proportion Surviving at End of Interval",
                            "Std. Error of Cumulative Proportion Surviving at End of Interval",
                            "Probability Density", "Std. Error of Probability Density",
                            "Hazard Rate", "Std. Error of Hazard Rate"]
        sec.table(Table("Life Table" + ("ᵃ" if True else ""), cols, rows,
                        row_header_cols=max(len(lead_cols), 1) if lead_cols else 0,
                        footnotes=[f"a. The median survival time is shown in the next table."]))
    sec.table(Table("Median Survival Times", (lead_cols or [""]) + ["Median Time"],
                    medians if lead_cols else [[""] + r for r in medians],
                    row_header_cols=max(len(lead_cols), 1)))
    if compare and factor and len(flevels) > 1:
        crow = []
        for bv in blevels:
            m = np.ones(len(fr), bool) if bv is None else (fr[by] == bv).to_numpy()
            sub = fr[m]
            lead = [_level_label(ds, by, bv)] if by else []
            if not pairwise:
                stat, df, p = lee_desu(sub["t"], sub["d"], sub["f"])
                crow.append(lead + ["Overall", _f(stat, 3), fmt_int(df), _sig(p)])
            else:
                for i, a in enumerate(flevels):
                    for b in flevels[i + 1:]:
                        mm = sub["f"].isin([a, b])
                        stat, df, p = lee_desu(sub["t"][mm], sub["d"][mm], sub["f"][mm])
                        crow.append(lead + [f"{_level_label(ds, factor, a)} vs "
                                            f"{_level_label(ds, factor, b)}",
                                            _f(stat, 3), fmt_int(df), _sig(p)])
        sec.table(Table("Overall Comparisonsᵃ" if not pairwise else "Pairwise Comparisonsᵃ",
                        ([ds.var(by).title] if by else []) + ["", "Wilcoxon (Gehan) Statistic",
                                                             "df", "Sig."],
                        crow, row_header_cols=2 if by else 1,
                        footnotes=["a. Comparisons are exact significance levels."]))
    names = {"survival": "Survival Function", "log_survival": "Log Survival Function",
             "hazard": "Hazard Function", "density": "Density Function",
             "one_minus": "One Minus Survival Function"}
    for kind in plots:
        for bv in blevels:
            fig = figure(6.2, 4.4)
            ax = fig.add_subplot(111)
            for i, fv in enumerate(flevels):
                lt = curves.get((bv, fv))
                if lt is None:
                    continue
                ends = lt["start"].to_numpy() + width
                if kind in ("survival", "log_survival", "one_minus"):
                    x = np.concatenate([[0.0], ends])
                    yv = np.concatenate([[1.0], lt["cum"].to_numpy()])
                    with np.errstate(divide="ignore"):
                        y = {"survival": yv, "log_survival": np.log(yv),
                             "one_minus": 1 - yv}[kind]
                    ax.step(x, y, where="post", color=COLORS[i % len(COLORS)],
                            label=_level_label(ds, factor, fv) if factor else None)
                else:
                    y = lt["hazard" if kind == "hazard" else "density"].to_numpy()
                    ax.plot(lt["start"].to_numpy(), y, marker="o", ms=3,
                            color=COLORS[i % len(COLORS)],
                            label=_level_label(ds, factor, fv) if factor else None)
            title = names[kind]
            if by:
                title += f" ({ds.var(by).title} = {_level_label(ds, by, bv)})"
            ax.set_title(title)
            ax.set_xlabel(tvar.title)
            ax.set_ylabel(names[kind])
            ax.grid(alpha=0.25)
            if factor:
                ax.legend(frameon=False, fontsize=8)
            sec.image(title, png(fig))
    return sec


# =========================================================================== Cox engine
class CoxData:
    """Breslow partial likelihood for (start, stop] counting-process data with strata."""

    def __init__(self, stop, event, X, strata=None, start=None):
        self.stop = np.asarray(stop, float)
        self.event = np.asarray(event, float)
        self.X = np.asarray(X, float)
        if self.X.ndim == 1:
            self.X = self.X[:, None]
        self.n = len(self.stop)
        self.start = None if start is None else np.asarray(start, float)
        st = np.zeros(self.n) if strata is None else np.asarray(strata, dtype=object)
        self.strata_values = st
        self.blocks = []
        for s in sorted(set(st.tolist()), key=sort_key):
            idx = np.nonzero(st == s)[0]
            stop_s = self.stop[idx]
            ev = self.event[idx] == 1
            et = np.unique(stop_s[ev])
            o_exit = idx[np.argsort(stop_s, kind="mergesort")]
            pos_exit = np.searchsorted(self.stop[o_exit], et, "left")
            if self.start is not None:
                o_entry = idx[np.argsort(self.start[idx], kind="mergesort")]
                pos_entry = np.searchsorted(self.start[o_entry], et, "left")
            else:
                o_entry = pos_entry = None
            te = np.sort(stop_s[ev])
            dcount = (np.searchsorted(te, et, "right") - np.searchsorted(te, et, "left"))
            self.blocks.append(dict(stratum=s, idx=idx, et=et, d=dcount.astype(float),
                                    o_exit=o_exit, pos_exit=pos_exit, o_entry=o_entry,
                                    pos_entry=pos_entry, ev_idx=idx[ev]))

    @staticmethod
    def _rev_cumsum(a):
        out = np.cumsum(a[::-1], axis=0)[::-1]
        return np.concatenate([out, np.zeros((1,) + a.shape[1:])], axis=0)

    def risk_sums(self, block, beta, cols=None, order: int = 2, X=None):
        X = self.X if X is None else X
        X = X if cols is None else X[:, cols]
        eta = X @ beta if X.shape[1] else np.zeros(self.n)
        shift = float(eta[block["idx"]].max()) if len(block["idx"]) else 0.0
        w = np.exp(eta - shift)

        def acc(values):
            ve = values[block["o_exit"]]
            total = self._rev_cumsum(ve)[block["pos_exit"]]
            if block["o_entry"] is not None:
                vn = values[block["o_entry"]]
                total = total - self._rev_cumsum(vn)[block["pos_entry"]]
            return total

        s0 = acc(w)
        s1 = acc(w[:, None] * X) if order >= 1 else None
        s2 = acc(w[:, None, None] * X[:, :, None] * X[:, None, :]) if order >= 2 else None
        return s0, s1, s2, shift, eta

    def loglik(self, beta, cols=None, order: int = 2):
        X = self.X if cols is None else self.X[:, cols]
        p = X.shape[1]
        ll = 0.0
        g = np.zeros(p)
        H = np.zeros((p, p))
        for b in self.blocks:
            if not len(b["et"]):
                continue
            s0, s1, s2, shift, eta = self.risk_sums(b, beta, cols, order)
            ll += eta[b["ev_idx"]].sum() - np.sum(b["d"] * (np.log(s0) + shift))
            if order >= 1 and p:
                xbar = s1 / s0[:, None]
                g += X[b["ev_idx"]].sum(0) - (b["d"][:, None] * xbar).sum(0)
                if order >= 2:
                    m2 = s2 / s0[:, None, None] - xbar[:, :, None] * xbar[:, None, :]
                    H -= (b["d"][:, None, None] * m2).sum(0)
        return ll, g, H


@dataclass
class CoxFit:
    cols: list[int]
    beta: np.ndarray
    cov: np.ndarray
    ll: float
    iterations: int
    converged: bool


def cox_fit(data: CoxData, cols: list[int], beta0=None, max_iter: int = 20,
            tol: float = 1e-9) -> CoxFit:
    p = len(cols)
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    if p == 0:
        ll, _, _ = data.loglik(beta, cols, order=0)
        return CoxFit(cols, beta, np.zeros((0, 0)), ll, 0, True)
    ll, g, H = data.loglik(beta, cols)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(-H) @ g
        factor = 1.0
        while True:
            new = beta + factor * step
            ll_new, g_new, H_new = data.loglik(new, cols)
            if ll_new >= ll - 1e-12 or factor < 1e-4:
                break
            factor /= 2
        change = abs(ll_new - ll)
        beta, ll, g, H = new, ll_new, g_new, H_new
        if change < tol * max(1.0, abs(ll)) and np.max(np.abs(factor * step)) < 1e-6:
            converged = True
            break
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(-H)
    return CoxFit(cols, beta, cov, ll, it, converged)


def score_statistic(data: CoxData, cols: list[int], beta: np.ndarray) -> float:
    _, g, H = data.loglik(beta, cols)
    try:
        return float(g @ np.linalg.solve(-H, g))
    except np.linalg.LinAlgError:
        return float(g @ np.linalg.pinv(-H) @ g)


# =========================================================================== Cox regression
@dataclass
class _Term:
    name: str
    title: str
    cols: list[int]
    coding: Coding | None = None


def episode_split(frame: pd.DataFrame, time_col: str = "t", event_col: str = "d",
                  strata_col: str | None = None) -> pd.DataFrame:
    """Split each case at every event time (of its stratum) up to its own time. Rows keep the
    original index in ``_case``; ``_start``/``_stop`` define (start, stop]; ``T_`` = stop."""
    pieces = []
    groups = [(None, frame)] if strata_col is None else list(frame.groupby(strata_col))
    for _, sub in groups:
        et = np.unique(sub.loc[sub[event_col] == 1, time_col].to_numpy(float))
        if not len(et):
            continue
        t = sub[time_col].to_numpy(float)
        counts = np.searchsorted(et, t, "right")
        rep = np.repeat(np.arange(len(sub)), counts)
        k = np.concatenate([np.arange(c) for c in counts]) if len(rep) else np.array([], int)
        stop = et[k]
        start = np.where(k > 0, et[np.maximum(k - 1, 0)], -np.inf)
        part = sub.iloc[rep].copy()
        part["_case"] = sub.index.to_numpy()[rep]
        part["_start"] = start
        part["_stop"] = stop
        part["_event"] = ((part[event_col].to_numpy() == 1) & (stop == part[time_col].to_numpy())
                          ).astype(float)
        pieces.append(part)
    if not pieces:
        return frame.iloc[:0].assign(_case=[], _start=[], _stop=[], _event=[])
    out = pd.concat(pieces)
    out.index = pd.RangeIndex(len(out))
    return out


def _evaluate_time_dependent(ds: Dataset, episodes: pd.DataFrame, expression: str) -> np.ndarray:
    from rsstatchem.core.expression import evaluate

    rows = episodes["_case"].to_numpy()
    df = ds.df.iloc[rows].reset_index(drop=True).copy()
    variables = [copy.deepcopy(v) for v in ds.variables if v.name.upper() != "T_"]
    df = df[[v.name for v in variables]]
    df["T_"] = episodes["_stop"].to_numpy()
    variables.append(Variable(name="T_", type=NUMERIC))
    ep = Dataset(df, variables)
    return pd.to_numeric(evaluate(ep, expression), errors="coerce").to_numpy(float)


def cox_regression(ds: Dataset, time: str, status: str, event, covariates: list[str],
                   categorical: dict | None = None, method: str = "enter",
                   strata: str | None = None, p_enter: float = 0.05, p_remove: float = 0.10,
                   max_iter: int = 20, ci: bool = True, level: float = 0.95,
                   correlations: bool = False, baseline: bool = False,
                   display_each_step: bool = True, plots: tuple = (),
                   separate_lines: str | None = None, save: tuple = (),
                   time_dependent: str | None = None) -> Result:
    """SPSS COXREG. ``categorical`` maps covariate -> (contrast, reference). ``plots`` from
    survival, hazard, lml, one_minus. ``save`` from survival, se, lml, hazard, xbeta, partial,
    dfbeta. ``time_dependent`` is an expression of ``T_`` that defines the covariate T_COV_
    (include "T_COV_" in ``covariates`` to use it)."""
    categorical = dict(categorical or {})
    base_covs = [c for c in covariates if c.upper() != "T_COV_"]
    use_tcov = time_dependent is not None and any(c.upper() == "T_COV_" for c in covariates)
    for name in base_covs:
        if ds.var(name).type != NUMERIC and name not in categorical:
            categorical[name] = ("indicator", "last")
    sd = _prepare(ds, time, status, event, None, strata, base_covs)
    fr = sd.frame
    sec = Section("Cox Regression" if not time_dependent else
                  "Cox Regression with Time-Dependent Covariates")
    syntax = ""
    if time_dependent:
        syntax += f"TIME PROGRAM.\nCOMPUTE T_COV_ = {time_dependent}.\n"
    syntax += f"COXREG {time}\n  /STATUS={status}{describe_event(event)}"
    if strata:
        syntax += f"\n  /STRATA={strata}"
    for name, (con, ref) in categorical.items():
        if name in base_covs:
            syntax += f"\n  /CONTRAST ({name})={con.capitalize()}" \
                      f"{'(1)' if ref == 'first' and con in _REF_CONTRASTS else ''}"
    syntax += f"\n  /METHOD={method.split('_')[0].upper()}" \
              f"{'' if method == 'enter' else ' ' + method.split('_')[1].upper()} " + \
              " ".join(covariates)
    syntax += f"\n  /CRITERIA=PIN({p_enter:g}) POUT({p_remove:g}) ITERATE({max_iter})."
    sec.log(syntax)

    # Build design
    names_x: list[str] = []
    blocks_x: list[np.ndarray] = []
    terms: list[_Term] = []
    codings: list[Coding] = []
    for name in base_covs:
        var = ds.var(name)
        if name in categorical:
            con, ref = categorical[name]
            coding = make_coding(ds, name, fr[name], con, ref)
            codings.append(coding)
            enc = coding.encode(fr[name]).to_numpy()
            cols = list(range(len(names_x), len(names_x) + enc.shape[1]))
            names_x += coding.columns
            blocks_x.append(enc)
            terms.append(_Term(name, var.title, cols, coding))
        else:
            cols = [len(names_x)]
            names_x.append(name)
            blocks_x.append(fr[name].to_numpy(float)[:, None])
            terms.append(_Term(name, name, cols))
    X = np.hstack(blocks_x) if blocks_x else np.zeros((len(fr), 0))
    keep = ~np.isnan(X).any(axis=1) if X.shape[1] else np.ones(len(fr), bool)
    fr = fr[keep]
    X = X[keep]
    n_missing = sd.n_missing + int((~keep).sum())

    # censored before the first event of its stratum
    st = fr["s"].to_numpy(dtype=object) if strata else np.zeros(len(fr))
    early = np.zeros(len(fr), bool)
    for s in set(st.tolist()):
        m = st == s
        ev_t = fr["t"].to_numpy()[m & (fr["d"].to_numpy() == 1)]
        first = ev_t.min() if len(ev_t) else np.inf
        early |= m & (fr["d"].to_numpy() == 0) & (fr["t"].to_numpy() < first)
    n_events = int(fr["d"].sum())
    n_cens = int(len(fr) - n_events - early.sum())
    total = sd.n_total
    rows = [["Cases available in analysis", "Eventᵃ", fmt_int(n_events),
             _f(100 * n_events / total, 1) + "%"],
            ["", "Censored", fmt_int(n_cens), _f(100 * n_cens / total, 1) + "%"],
            ["", "Total", fmt_int(n_events + n_cens),
             _f(100 * (n_events + n_cens) / total, 1) + "%"],
            ["Cases dropped", "Cases with missing values", fmt_int(n_missing),
             _f(100 * n_missing / total, 1) + "%"],
            ["", "Cases with negative time", fmt_int(sd.n_negative),
             _f(100 * sd.n_negative / total, 1) + "%"],
            ["", "Censored cases before the earliest event in a stratum",
             fmt_int(int(early.sum())), _f(100 * early.sum() / total, 1) + "%"],
            ["", "Total", fmt_int(n_missing + sd.n_negative + int(early.sum())),
             _f(100 * (n_missing + sd.n_negative + early.sum()) / total, 1) + "%"],
            ["Total", "", fmt_int(total), "100.0%"]]
    sec.table(Table("Case Processing Summary", ["", "", "N", "Percent"], rows,
                    row_header_cols=2, footnotes=[f"a. Dependent Variable: {ds.var(time).title}"]))
    if n_events == 0:
        sec.note("Warning", "There are no events: the model cannot be estimated.", "warning")
        return Result(sec)

    if codings:
        crow = []
        for c in codings:
            counts = fr[c.name].value_counts()
            for i, (cat, lab) in enumerate(zip(c.categories, c.labels)):
                crow.append([c.name if i == 0 else "", lab, fmt_int(counts.get(cat, 0))] +
                            [fmt(v, 3) if c.contrast not in ("indicator", "deviation") else
                             fmt_int(v) for v in c.matrix[i]])
        width = max(c.matrix.shape[1] for c in codings)
        crow = [r + [""] * (3 + width - len(r)) for r in crow]
        notes = [f"{c.name}: {CONTRASTS[c.contrast]} parameter coding"
                 f"{' (reference ' + c.reference + ')' if c.contrast in _REF_CONTRASTS else ''}"
                 for c in codings]
        sec.table(Table("Categorical Variable Codings", ["", "", "Frequency"] +
                        [f"({j + 1})" for j in range(width)], crow, row_header_cols=2,
                        footnotes=notes))

    # Time-dependent covariate: episode split
    tcov_index = None
    if use_tcov:
        work = fr.copy()
        work["_x_index"] = np.arange(len(fr))
        ep = episode_split(work, "t", "d", "s" if strata else None)
        tc = _evaluate_time_dependent(ds, ep, time_dependent)
        good = ~np.isnan(tc)
        ep = ep[good]
        tc = tc[good]
        Xe = np.hstack([X[ep["_x_index"].to_numpy()], tc[:, None]])
        tcov_index = len(names_x)
        names_x.append("T_COV_")
        terms.insert([c.upper() for c in covariates].index("T_COV_"),
                     _Term("T_COV_", "T_COV_", [tcov_index]))
        data = CoxData(ep["_stop"], ep["_event"], Xe,
                       ep["s"].to_numpy(dtype=object) if strata else None, ep["_start"])
    else:
        data = CoxData(fr["t"], fr["d"], X, st if strata else None)

    alpha_z = stats.norm.ppf(0.5 + level / 2)
    null = cox_fit(data, [])
    sec.table(Table("Omnibus Tests of Model Coefficients (Block 0: Beginning Block)",
                    ["", "-2 Log Likelihood"], [["", _f(-2 * null.ll, 3)]]))

    steps = _cox_select(data, terms, method, p_enter, p_remove, max_iter)
    shown = set(range(len(steps))) if display_each_step else {len(steps) - 1}
    prev_ll = null.ll
    omni_rows = []
    for step_no, (in_terms, fit, action) in enumerate(steps):
        cols_all = [c for t in in_terms for c in t.cols]
        score = score_statistic(data, cols_all, np.zeros(len(cols_all))) if cols_all else 0.0
        df_all = len(cols_all)
        chg = 2 * (fit.ll - prev_ll)
        df_chg = df_all - (len([c for t in steps[step_no - 1][0] for c in t.cols])
                           if step_no else 0)
        blk = 2 * (fit.ll - null.ll)
        label = f"{step_no + 1}ᵃ" if method != "enter" else ""
        if method == "enter" or step_no > 0 or action:
            omni_rows.append([label, _f(-2 * fit.ll, 3), _f(score, 3), fmt_int(df_all),
                              _sig(stats.chi2.sf(score, df_all)) if df_all else "",
                              _f(chg, 3), fmt_int(abs(df_chg)),
                              _sig(stats.chi2.sf(abs(chg), abs(df_chg))) if df_chg else "",
                              _f(blk, 3), fmt_int(df_all),
                              _sig(stats.chi2.sf(blk, df_all)) if df_all else ""])
        prev_ll = fit.ll
    foot = [f"Beginning Block Number 1. Method = {COX_METHODS[method]}"]
    if method != "enter":
        actions = [a for _, _, a in steps if a]
        foot += [f"{sup_letter(i)}. {a}" for i, a in enumerate(actions)]
    sec.table(Table("Omnibus Tests of Model Coefficients",
                    ["Step" if method != "enter" else "", "-2 Log Likelihood",
                     ("Overall (score)", "Chi-square"), ("Overall (score)", "df"),
                     ("Overall (score)", "Sig."),
                     ("Change From Previous Step", "Chi-square"),
                     ("Change From Previous Step", "df"),
                     ("Change From Previous Step", "Sig."),
                     ("Change From Previous Block", "Chi-square"),
                     ("Change From Previous Block", "df"),
                     ("Change From Previous Block", "Sig.")],
                    omni_rows or [["", _f(-2 * null.ll, 3)] + [""] * 9], footnotes=foot))

    ci_head = f"{level * 100:g}% CI for Exp(B)"
    for step_no, (in_terms, fit, _a) in enumerate(steps):
        if step_no not in shown:
            continue
        step_label = f"Step {step_no + 1}" if method != "enter" else ""
        vrows = []
        pos = {c: i for i, c in enumerate(fit.cols)}
        for term in in_terms:
            idx = [pos[c] for c in term.cols]
            if term.coding is not None:
                b = fit.beta[idx]
                v = fit.cov[np.ix_(idx, idx)]
                wald = float(b @ np.linalg.pinv(v) @ b)
                vrows.append([step_label if not vrows else "", term.name, "", "", _f(wald, 3),
                              fmt_int(len(idx)), _sig(stats.chi2.sf(wald, len(idx))), ""] +
                             (["", ""] if ci else []))
            for c, i in zip(term.cols, idx):
                b = fit.beta[i]
                se = math.sqrt(max(fit.cov[i, i], 0))
                wald = (b / se) ** 2 if se > 0 else float("nan")
                row = [step_label if not vrows else "", names_x[c], _f(b, 3), _f(se, 3),
                       _f(wald, 3), "1", _sig(stats.chi2.sf(wald, 1)), _f(math.exp(b), 3)]
                if ci:
                    row += [_f(math.exp(b - alpha_z * se), 3), _f(math.exp(b + alpha_z * se), 3)]
                vrows.append(row)
        if vrows:
            cols = ["", "", "B", "SE", "Wald", "df", "Sig.", "Exp(B)"]
            if ci:
                cols += [(ci_head, "Lower"), (ci_head, "Upper")]
            sec.table(Table("Variables in the Equation", cols, vrows, row_header_cols=2))
        out_terms = [t for t in terms if t not in in_terms]
        if method != "enter" and out_terms:
            cols_in = [c for t in in_terms for c in t.cols]
            beta_full_cols = cols_in
            nrows = []
            for t in out_terms:
                cols = beta_full_cols + t.cols
                beta = np.concatenate([fit.beta, np.zeros(len(t.cols))])
                sc = score_statistic(data, cols, beta)
                nrows.append([step_label if not nrows else "", t.name, _f(sc, 3),
                              fmt_int(len(t.cols)), _sig(stats.chi2.sf(sc, len(t.cols)))])
            all_out = [c for t in out_terms for c in t.cols]
            beta = np.concatenate([fit.beta, np.zeros(len(all_out))])
            rs = score_statistic(data, cols_in + all_out, beta)
            sec.table(Table("Variables not in the Equationᵃ", ["", "", "Score", "df", "Sig."],
                            nrows, row_header_cols=2,
                            footnotes=[f"a. Residual Chi Square = {_f(rs, 3)} with "
                                       f"{len(all_out)} df Sig. = "
                                       f"{_sig(stats.chi2.sf(rs, len(all_out)))}"]))

    final_terms, final, _ = steps[-1]
    if correlations and len(final.cols) > 1:
        sdv = np.sqrt(np.diag(final.cov))
        corr = final.cov / np.outer(sdv, sdv)
        labels = [names_x[c] for c in final.cols]
        sec.table(Table("Correlation Matrix of Regression Coefficients", [""] + labels,
                        [[labels[i]] + [_f(corr[i, j], 3) for j in range(len(labels))]
                         for i in range(len(labels))]))

    # covariate means over the cases in the analysis (SPSS)
    means_all = X.mean(axis=0) if X.shape[1] else np.zeros(0)
    if tcov_index is not None:
        means_all = np.concatenate([means_all, [float(np.nanmean(data.X[:, tcov_index]))]])
    if len(names_x):
        sec.table(Table("Covariate Means", ["", "Mean"],
                        [[names_x[i], _f(means_all[i], 3)] for i in range(len(names_x))]))

    cols = final.cols
    beta = final.beta
    models = []
    if len(cols):
        from rsstatchem.core.stats import boot_models as bm
        info = bm.cox_model(fr["t"].to_numpy(float), fr["d"].to_numpy(float),
                            data.X[:, cols] if tcov_index is None else X[:, cols[:0]],
                            st if strata else None, [names_x[c] for c in cols], beta,
                            se=np.sqrt(np.clip(np.diag(final.cov), 0, None)),
                            rows=fr.index.to_numpy(), table="Variables in the Equation",
                            headers=[""], stat="wald", exp=True, stat_label="Wald")
        if tcov_index is not None:
            info.bootable = False
            info.note = ("Bootstrap results are not available for models with time-dependent "
                         "covariates.")
        elif method != "enter":
            info.note = ("Variable selection is not repeated in the bootstrap samples: the "
                         "final model is re-estimated with the selected covariates.")
        models.append(info)
    zbar = means_all[cols] if len(cols) else np.zeros(0)
    base = _baseline(data, cols, beta, final.cov, zbar)
    if baseline and tcov_index is None:
        brow = []
        for sname, b in base.items():
            for k in range(len(b["t"])):
                brow.append(([_level_label(ds, strata, sname) if k == 0 else ""] if strata
                             else []) + [fmt(b["t"][k], 3), _f(b["H0"][k], 3),
                                         _f(b["S"][k], 3), _f(b["se"][k], 3), _f(b["H"][k], 3)])
        sec.table(Table("Survival Table",
                        ([ds.var(strata).title] if strata else []) +
                        ["Time", "Baseline Cum Hazard", ("At mean of covariates", "Survival"),
                         ("At mean of covariates", "SE"),
                         ("At mean of covariates", "Cumulative Hazard")], brow,
                        row_header_cols=2 if strata else 1))

    if plots and tcov_index is None:
        _cox_plots(sec, ds, base, strata, plots, separate_lines, final_terms, names_x, cols,
                   beta, zbar, time)
    elif plots:
        sec.note("", "Plots are not available when the model has time-dependent covariates.")

    saved = {}
    if save and tcov_index is None:
        saved = _cox_save(ds, fr, data, base, cols, beta, final.cov, means_all, names_x,
                          save, strata)
    elif save:
        sec.note("", "Variables cannot be saved for models with time-dependent covariates.")
    if not final.converged:
        sec.note("Warning", "Estimation terminated because the maximum number of iterations "
                 "was reached; final solution cannot be found.", "warning")
    return Result(sec, saved, models)


def sup_letter(i: int) -> str:
    from rsstatchem.core.stats.common import sup

    return sup("abcdefghijklmnoprstuvwxyz"[i % 25])


def _cox_select(data: CoxData, terms: list[_Term], method: str, p_enter: float,
                p_remove: float, max_iter: int) -> list[tuple[list[_Term], CoxFit, str]]:
    """Returns the model after each step: (terms in model, fit, description of the step)."""
    def fit_terms(ts, start=None):
        cols = [c for t in ts for c in t.cols]
        return cox_fit(data, cols, start, max_iter)

    if method == "enter":
        return [(list(terms), fit_terms(terms), "")]
    kind, crit = method.split("_")
    current = [] if kind == "fstep" else list(terms)
    fit = fit_terms(current)
    steps = [] if kind == "fstep" else [(list(current), fit, "")]
    seen = set()

    def try_enter():
        nonlocal current, fit
        out = [t for t in terms if t not in current]
        best, best_p = None, 1.0
        cols_in = [c for t in current for c in t.cols]
        for t in out:
            beta = np.concatenate([fit.beta, np.zeros(len(t.cols))])
            sc = score_statistic(data, cols_in + t.cols, beta)
            p = stats.chi2.sf(sc, len(t.cols))
            if p < best_p:
                best, best_p = t, p
        if best is not None and best_p < p_enter:
            key = ("in", best.name, len(current))
            if key in seen:
                return False
            seen.add(key)
            current = current + [best]
            current.sort(key=lambda t: terms.index(t))
            fit = fit_terms(current)
            steps.append((list(current), fit, f"Variable(s) Entered at Step Number "
                                              f"{len(steps) + 1}: {best.name}"))
            return True
        return False

    def removal_stat(t):
        cols_in = [c for x in current for c in x.cols]
        pos = [cols_in.index(c) for c in t.cols]
        if crit == "wald":
            b = fit.beta[pos]
            v = fit.cov[np.ix_(pos, pos)]
            return float(b @ np.linalg.pinv(v) @ b)
        rest = [x for x in current if x is not t]
        if crit == "lr":
            return 2 * (fit.ll - fit_terms(rest).ll)
        # conditional: one Newton step from the current estimates with the term removed
        keep = [i for i in range(len(cols_in)) if i not in pos]
        cols = [cols_in[i] for i in keep]
        b0 = fit.beta[keep]
        if not cols:
            ll0, _, _ = data.loglik(np.zeros(0), [], order=0)
            return 2 * (fit.ll - ll0)
        _, g, H = data.loglik(b0, cols)
        try:
            b1 = b0 + np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            b1 = b0
        ll1, _, _ = data.loglik(b1, cols, order=0)
        return 2 * (fit.ll - ll1)

    def try_remove():
        nonlocal current, fit
        if not current:
            return False
        worst, worst_p = None, -1.0
        for t in current:
            p = stats.chi2.sf(max(removal_stat(t), 0.0), len(t.cols))
            if p > worst_p:
                worst, worst_p = t, p
        if worst is not None and worst_p > p_remove:
            key = ("out", worst.name, len(current))
            if key in seen:
                return False
            seen.add(key)
            current = [t for t in current if t is not worst]
            fit = fit_terms(current)
            steps.append((list(current), fit, f"Variable Removed at Step Number "
                                              f"{len(steps) + 1}: {worst.name}"))
            return True
        return False

    for _ in range(4 * len(terms) + 4):
        if kind == "fstep":
            moved = try_enter()
            if moved:
                while try_remove():
                    pass
                continue
            break
        moved = try_remove()
        if moved:
            continue
        if not try_enter():
            break
    if not steps:
        steps.append((list(current), fit, ""))
    return steps


def _kp_alpha(e_dead: np.ndarray, risk: float) -> float:
    """Kalbfleisch-Prentice conditional survival alpha_i at one event time (IBM COXREG
    Algorithms, "Estimation of the Baseline Function"): closed form for a single event,
    otherwise the root of sum_D e/(1 - alpha^e) = sum_R e, started from Breslow's
    exp(-d / sum_R e). ``e_dead`` are exp(x'b) of the events (centred x), ``risk`` the risk-set
    sum of exp(x'b)."""
    if len(e_dead) == 1:
        e = float(e_dead[0])
        base = 1 - e / risk
        return float(base ** (1 / e)) if base > 0 else 0.0

    def g(a):
        return float(np.sum(e_dead / (1 - a ** e_dead))) - risk
    if float(np.sum(e_dead)) >= risk * (1 - 1e-12):   # everybody at risk fails
        return 0.0
    # g increases from sum_D e - sum_R e < 0 (alpha -> 0) to +inf (alpha -> 1)
    lo, hi = 0.0, 1 - 1e-15
    while g(hi) < 0 and hi < 1 - 1e-300:
        hi = 1 - (1 - hi) / 1e3
    return float(optimize.brentq(g, lo, hi, xtol=1e-15, rtol=1e-14))


def _baseline(data: CoxData, cols, beta, cov, zbar, estimator: str = "kp") -> dict:
    """Baseline cumulative hazard (covariates = 0) and survival at the covariate means, per
    stratum, with the SE of the survival at the means.

    ``estimator`` "kp" (default, IBM COXREG: covariates centred at their means and the
    Kalbfleisch-Prentice product S0(t) = prod alpha_i, iterative for tied events) or
    "breslow" (h0 = d / sum exp(x'b)). The variance of -ln S0(t) is the IBM / Kalbfleisch-
    Prentice formula sum |D_i| (sum_R exp(x'b))^-2 + a' I^-1 a in both cases."""
    out = {}
    X = data.X[:, cols] if len(cols) else np.zeros((data.n, 0))
    for b in data.blocks:
        if not len(b["et"]):
            continue
        s0, s1, _, shift, eta = data.risk_sums(b, beta, cols, order=1)
        s0_true = s0 * math.exp(shift)
        rel = math.exp(float(zbar @ beta)) if len(cols) else 1.0
        h0 = b["d"] / s0_true
        if estimator == "kp":
            risk_c = s0_true / rel                     # sum over R of exp((x - xbar)'b)
            ev = b["ev_idx"]
            e_c = np.exp(eta[ev] - float(zbar @ beta)) if len(cols) else np.ones(len(ev))
            pos = np.searchsorted(b["et"], data.stop[ev])
            alpha = np.array([_kp_alpha(e_c[pos == i], float(risk_c[i]))
                              for i in range(len(b["et"]))])
            with np.errstate(divide="ignore"):
                H = -np.log(np.cumprod(alpha))
            H0 = H / rel
        else:
            H0 = np.cumsum(h0)
            H = H0 * rel
        if len(cols):
            xbar = s1 / s0[:, None]
            q = np.cumsum((h0 * rel)[:, None] * (xbar - zbar), axis=0)
            var = np.cumsum(b["d"] * (rel / s0_true) ** 2) + np.einsum("ij,jk,ik->i", q, cov, q)
        else:
            var = np.cumsum(b["d"] / s0_true ** 2)
        S = np.exp(-H)
        out[b["stratum"]] = dict(t=b["et"], H0=H0, H=H, S=S, se=S * np.sqrt(var))
    _ = X
    return out


def _cox_plots(sec, ds, base, strata, plots, separate_lines, terms, names_x, cols, beta,
               zbar, time) -> None:
    patterns = [("", zbar)]
    if separate_lines:
        term = next((t for t in terms if t.name == separate_lines and t.coding is not None),
                    None)
        if term is not None:
            patterns = []
            for lab, row in zip(term.coding.labels, term.coding.matrix):
                z = zbar.copy()
                for c, v in zip(term.cols, row):
                    if c in cols:
                        z[cols.index(c)] = v
                patterns.append((lab, z))
    titles = {"survival": "Survival Function", "hazard": "Hazard Function",
              "lml": "LML Function", "one_minus": "One Minus Survival Function"}
    for kind in plots:
        fig = figure(6.2, 4.4)
        ax = fig.add_subplot(111)
        i = 0
        for sname, b in base.items():
            for lab, z in patterns:
                rel = math.exp(float(z @ beta)) if len(cols) else 1.0
                H = b["H0"] * rel
                S = np.exp(-H)
                with np.errstate(divide="ignore"):
                    y = {"survival": S, "hazard": H, "lml": np.log(H), "one_minus": 1 - S}[kind]
                start = {"survival": 1.0, "hazard": 0.0, "lml": np.nan, "one_minus": 0.0}[kind]
                x = np.concatenate([[0.0], b["t"]])
                yy = np.concatenate([[start], y])
                parts = [p for p in (_level_label(ds, strata, sname) if strata else "", lab) if p]
                ax.step(x, yy, where="post", color=COLORS[i % len(COLORS)],
                        label=" / ".join(parts) or None)
                i += 1
        title = titles[kind] + (" at mean of covariates" if not separate_lines else
                                f" for patterns 1 - {len(patterns)}")
        ax.set_title(title)
        ax.set_xlabel(ds.var(time).title)
        ax.set_ylabel({"survival": "Cum Survival", "hazard": "Cum Hazard",
                       "lml": "Log minus log", "one_minus": "1 - Cum Survival"}[kind])
        ax.grid(alpha=0.25)
        if i > 1:
            ax.legend(frameon=False, fontsize=8)
        sec.image(title, png(fig))


def _cox_save(ds, fr, data: CoxData, base, cols, beta, cov, means_all, names_x, save,
              strata) -> dict:
    index = ds.df.index
    saved = {}
    X = data.X[:, cols] if len(cols) else np.zeros((data.n, 0))
    eta = X @ beta if len(cols) else np.zeros(data.n)
    zbar = means_all[cols] if len(cols) else np.zeros(0)
    t = fr["t"].to_numpy()
    st = fr["s"].to_numpy(dtype=object) if strata else np.zeros(len(fr))
    H = np.full(len(fr), np.nan)
    se = np.full(len(fr), np.nan)
    for sname, b in base.items():
        m = st == sname
        k = np.searchsorted(b["t"], t[m], "right") - 1
        H0 = np.where(k >= 0, b["H0"][np.clip(k, 0, None)], 0.0)
        H[m] = H0 * np.exp(eta[m])
        # SE of S(t|x_i) by the same delta method as at the means
        block = next(bl for bl in data.blocks if bl["stratum"] == sname)
        s0, s1, _, shift, _ = data.risk_sums(block, beta, cols, order=1)
        s0_true = s0 * math.exp(shift)
        h0 = block["d"] / s0_true
        xbar = s1 / s0[:, None] if len(cols) else np.zeros((len(h0), 0))
        cum_h0 = np.cumsum(h0)
        cum_hx = np.cumsum(h0[:, None] * xbar, axis=0) if len(cols) else None
        cum_v = np.cumsum(block["d"] / s0_true ** 2)
        idx = np.nonzero(m)[0]
        for j, kk in zip(idx, k):
            if kk < 0:
                se[j] = 0.0
                continue
            rel = math.exp(eta[j])
            var = cum_v[kk] * rel ** 2
            if len(cols):
                q = rel * (cum_hx[kk] - X[j] * cum_h0[kk])
                var += float(q @ cov @ q)
            se[j] = math.exp(-H[j]) * math.sqrt(max(var, 0))
    S = np.exp(-H)

    def full(values):
        s = pd.Series(np.nan, index=index)
        s.loc[fr.index] = values
        return s

    if "survival" in save:
        saved["SUR"] = ("Survival Function", full(S), 5)
    if "se" in save:
        saved["SE"] = ("Standard Error of Survival Function", full(se), 5)
    if "lml" in save:
        with np.errstate(divide="ignore"):
            saved["LML"] = ("Log Minus Log Survival Function",
                            full(np.where(H > 0, np.log(H), np.nan)), 5)
    if "hazard" in save:
        saved["HAZ"] = ("Cumulative Hazard Function", full(H), 5)
    if "xbeta" in save and len(cols):
        saved["XBE"] = ("X*Beta (mean-centered covariates)", full((X - zbar) @ beta), 5)
    if ("partial" in save or "dfbeta" in save) and len(cols):
        schoen, score_res = _residuals(data, cols, beta)
        if "partial" in save:
            for j, c in enumerate(cols):
                saved[f"PR{j + 1}"] = (f"Partial residual for {names_x[c]}",
                                       full(schoen[:, j]), 5)
        if "dfbeta" in save:
            dfb = score_res @ cov
            for j, c in enumerate(cols):
                saved[f"DFB{j + 1}"] = (f"DfBeta for {names_x[c]}", full(dfb[:, j]), 5)
    return saved


def _residuals(data: CoxData, cols, beta) -> tuple[np.ndarray, np.ndarray]:
    """Schoenfeld (partial) residuals for events (NaN for censored) and score residuals."""
    X = data.X[:, cols]
    p = len(cols)
    schoen = np.full((data.n, p), np.nan)
    score = np.zeros((data.n, p))
    eta = X @ beta
    for b in data.blocks:
        if not len(b["et"]):
            continue
        s0, s1, _, shift, _ = data.risk_sums(b, beta, cols, order=1)
        xbar = s1 / s0[:, None]
        s0_true = s0 * math.exp(shift)
        idx = b["idx"]
        t_i = data.stop[idx]
        k = np.searchsorted(b["et"], t_i, "left")
        ev = data.event[idx] == 1
        kk = np.clip(k, 0, len(b["et"]) - 1)
        schoen[idx[ev]] = X[idx[ev]] - xbar[kk[ev]]
        dh = b["d"] / s0_true
        cum_dh = np.cumsum(dh)
        cum_dhx = np.cumsum(dh[:, None] * xbar, axis=0)
        last = np.searchsorted(b["et"], t_i, "right") - 1
        rel = np.exp(eta[idx])
        ch = np.where(last >= 0, cum_dh[np.clip(last, 0, None)], 0.0)
        chx = np.where((last >= 0)[:, None], cum_dhx[np.clip(last, 0, None)], 0.0)
        sc = -rel[:, None] * (X[idx] * ch[:, None] - chx)
        sc[ev] += X[idx[ev]] - xbar[kk[ev]]
        score[idx] = sc
    return schoen, score

"""Analyze > Descriptive Statistics > Explore (SPSS EXAMINE).

Case processing summary, descriptives with a confidence interval, M-estimators (Huber,
Tukey's biweight, Hampel, Andrews' wave), percentiles (weighted average and Tukey's hinges),
extreme values (five highest and lowest cases, optionally labelled), tests of normality,
spread-versus-level with Levene's test, and the plots: boxplots (factor levels together or
dependents together), stem-and-leaf, histogram, normal and detrended normal Q-Q plots.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import charts
from rsstatchem.core.stats.common import (Group, fmt, fmt_int, fmt_p, groups_of, lilliefors,
                                          numeric, pct, percentile, shapiro, t_interval,
                                          trimmed_mean, value_text)
from rsstatchem.core.stats.descriptive import summary

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
SPREAD_POWERS = {"log": 0.0, "inv_sqrt": -0.5, "reciprocal": -1.0, "sqrt": 0.5,
                 "square": 2.0, "cube": 3.0}
_POWER_TEXT = {0.0: "Natural log", -0.5: "1/square root", -1.0: "Reciprocal",
               0.5: "Square root", 2.0: "Square", 3.0: "Cube", 1.0: "Untransformed"}


# ============================================================================ pure helpers
def m_estimator(x: np.ndarray, kind: str = "huber", tol: float = 1e-10,
                max_iter: int = 500) -> float:
    """SPSS EXAMINE M-estimators of location: iteratively reweighted mean starting at the
    median with the scale fixed at MAD / 0.6745. Constants: Huber 1.339, Tukey biweight 4.685,
    Hampel (1.7, 3.4, 8.5), Andrews' wave 1.339 (cutoff 1.339*pi)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan")
    t = float(np.median(x))
    s = float(np.median(np.abs(x - t))) / 0.6744897501960817
    if s == 0:
        return t
    for _ in range(max_iter):
        u = (x - t) / s
        au = np.abs(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "huber":
                w = np.where(au <= 1.339, 1.0, 1.339 / au)
            elif kind == "tukey":
                w = np.where(au <= 4.685, (1 - (u / 4.685) ** 2) ** 2, 0.0)
            elif kind == "hampel":
                a, b, c = 1.7, 3.4, 8.5
                psi = np.where(au <= a, au, np.where(au <= b, a, np.where(
                    au <= c, a * (c - au) / (c - b), 0.0)))
                w = np.where(au > 0, psi / au, 1.0)
            else:  # andrews
                c = 1.339
                w = np.where(au <= c * math.pi,
                             np.where(au > 0, np.sin(au / c) / (au / c), 1.0), 0.0)
        if w.sum() <= 0:
            return float("nan")
        new = float((w * x).sum() / w.sum())
        if abs(new - t) <= tol * max(1.0, abs(t)):
            return new
        t = new
    return t


def weighted_average_percentiles(x: np.ndarray, ps=PERCENTILES) -> list[float]:
    return [float(v) for v in percentile(x, list(ps))]


def tukey_hinges(x: np.ndarray) -> tuple[float, float, float]:
    """Tukey's hinges as SPSS EXAMINE: depth of hinge = (floor(depth of median) + 1) / 2."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    def at(depth: float, from_top: bool = False) -> float:
        lo, hi = int(math.floor(depth)), int(math.ceil(depth))
        arr = x[::-1] if from_top else x
        return (arr[lo - 1] + arr[hi - 1]) / 2

    dm = (n + 1) / 2
    dh = (math.floor(dm) + 1) / 2
    return at(dh), float(np.median(x)), at(dh, from_top=True)


def extreme_values(x: np.ndarray, case_numbers: np.ndarray, k: int = 5) -> dict:
    """Five highest/lowest values with case numbers; ``partial`` flags a tie cut at the edge."""
    order_hi = np.lexsort((case_numbers, -x))
    order_lo = np.lexsort((case_numbers, x))
    out = {}
    for key, order in (("high", order_hi), ("low", order_lo)):
        chosen = order[:k]
        partial = None
        if len(order) > k and x[order[k]] == x[order[k - 1]]:
            partial = float(x[order[k - 1]])
        out[key] = [(int(case_numbers[i]), float(x[i]), int(i)) for i in chosen]
        out[key + "_partial"] = partial
    return out


def levene_examine(samples: list[np.ndarray]) -> list[tuple[str, float, float, float, float]]:
    """Levene's test as EXAMINE prints it, incl. the median version with adjusted df."""
    samples = [np.asarray(s, float) for s in samples if len(s) > 0]
    k = len(samples)
    n = sum(len(s) for s in samples)
    if k < 2 or n - k < 1:
        return []
    rows = []
    for title, center in (("Based on Mean", np.mean), ("Based on Median", np.median),
                          ("Based on Median and with adjusted df", np.median),
                          ("Based on trimmed mean", trimmed_mean)):
        z = [np.abs(s - center(s)) for s in samples]
        if all(np.ptp(d) == 0 for d in z) and len({float(d[0]) for d in z}) == 1:
            rows.append((title, float("nan"), k - 1, n - k, float("nan")))
            continue
        f, p = stats.f_oneway(*z)
        df2 = float(n - k)
        if "adjusted" in title:
            u = np.array([((d - d.mean()) ** 2).sum() for d in z])
            ni = np.array([len(d) for d in z], float)
            den = (u ** 2 / np.where(ni > 1, ni - 1, np.inf)).sum()
            df2 = float(u.sum() ** 2 / den) if den > 0 else float("nan")
            p = float(stats.f.sf(f, k - 1, df2))
        rows.append((title, float(f), k - 1, df2, float(p)))
    return rows


def stem_and_leaf(x: np.ndarray) -> dict:
    """Tukey stem-and-leaf display with SPSS-style extremes (beyond 1.5 IQR from the hinges)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return {"lines": [], "width": 1.0, "per_leaf": 1}
    lo_h, _, hi_h = tukey_hinges(x)
    step = 1.5 * (hi_h - lo_h)
    low_fence, high_fence = lo_h - step, hi_h + step
    low_ext = x[x < low_fence] if step > 0 else x[:0]
    high_ext = x[x > high_fence] if step > 0 else x[:0]
    body = x[(x >= low_fence) & (x <= high_fence)] if step > 0 else x
    spread = float(body.max() - body.min()) if len(body) else 0.0
    width = 10.0 ** math.floor(math.log10(spread)) if spread > 0 else \
        10.0 ** math.floor(math.log10(abs(body[0]))) if len(body) and body[0] != 0 else 1.0
    lines_per = 1
    if spread > 0:
        stems = spread / width
        lines_per = 5 if stems < 2 else 2 if stems < 5 else 1
    per_leaf = max(1, int(math.ceil(n / 300)))
    leaf_unit = width / 10
    scaled = np.floor(np.round(body / leaf_unit, 8)).astype(int)
    stem_vals = np.floor_divide(scaled, 10)
    leaf_vals = scaled - stem_vals * 10
    sub = leaf_vals // (10 // lines_per)
    lines = []
    if len(low_ext):
        lines.append((len(low_ext), "Extremes", f"(=<{_short(low_ext.max())})"))
    if len(body):
        keys = sorted(set(zip(stem_vals.tolist(), sub.tolist())))
        first, last = keys[0], keys[-1]
        cur = first
        while True:
            mask = (stem_vals == cur[0]) & (sub == cur[1])
            leaves = "".join(str(v) for v in sorted(leaf_vals[mask]))
            leaves = leaves[::per_leaf] if per_leaf > 1 else leaves
            lines.append((int(mask.sum()), str(cur[0]), leaves))
            if cur == last:
                break
            nxt = cur[1] + 1
            cur = (cur[0], nxt) if nxt < lines_per else (cur[0] + 1, 0)
    if len(high_ext):
        lines.append((len(high_ext), "Extremes", f"(>={_short(high_ext.min())})"))
    return {"lines": lines, "width": width, "per_leaf": per_leaf}


def _short(v: float) -> str:
    return f"{v:g}"


def spread_level(samples: list[np.ndarray], power: float | None = None) -> dict:
    """Spread-vs-level: ln(IQR) against ln(median); suggested power = 1 - slope."""
    meds, iqrs = [], []
    for s in samples:
        if power is not None and power != 1.0:
            s = _transform(s, power)
        if len(s) < 2:
            continue
        q1, q3 = percentile(s, [25, 75])
        med = float(np.median(s))
        if q3 - q1 > 0 and med > 0:
            meds.append(math.log(med))
            iqrs.append(math.log(q3 - q1))
    out = {"x": np.array(meds), "y": np.array(iqrs), "slope": float("nan"),
           "power": float("nan")}
    if len(meds) >= 2 and np.ptp(meds) > 0:
        slope = float(np.polyfit(meds, iqrs, 1)[0])
        out.update(slope=slope, power=1 - slope)
    return out


def _transform(x: np.ndarray, power: float) -> np.ndarray:
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(x) if power == 0 else np.power(x, power)
    return y[np.isfinite(y)]


# ============================================================================ procedure
def explore(ds: Dataset, dependents: list[str], factors: list[str] | None = None,
            statistics: bool = True, normality: bool = True, boxplot: bool = True,
            qq: bool = True, histogram: bool = False, level: float = 0.95, *,
            descriptives: bool = True, m_estimators: bool = False, outliers: bool = False,
            percentiles: bool = False, label_by: str | None = None,
            boxplot_mode: str | None = None, stem_leaf: bool = False,
            spread: str = "none", spread_power: str = "log", missing: str = "listwise"
            ) -> Section:
    """EXAMINE. ``boxplot_mode`` "factor" | "dependents" | "none" (``boxplot`` kept for the
    older call style); ``spread`` "none" | "estimate" | "transformed" | "untransformed";
    ``missing`` "listwise" | "pairwise" | "report"."""
    factors = factors or []
    if boxplot_mode is None:
        boxplot_mode = "factor" if boxplot else "none"
    section = Section("Explore")
    log = f"EXAMINE VARIABLES={' '.join(dependents)}"
    if factors:
        log += f" BY {' '.join(factors)}"
    if label_by:
        log += f"\n  /ID={label_by}"
    plot_kw = [k for k, on in (("BOXPLOT", boxplot_mode != "none"), ("STEMLEAF", stem_leaf),
                               ("HISTOGRAM", histogram), ("NPPLOT", normality)) if on]
    if spread != "none":
        plot_kw.append({"estimate": "SPREADLEVEL", "untransformed": "SPREADLEVEL(1)"}.get(
            spread, f"SPREADLEVEL({SPREAD_POWERS.get(spread_power, 0):g})"))
    log += f"\n  /PLOT {' '.join(plot_kw) or 'NONE'}"
    if boxplot_mode == "dependents":
        log += "\n  /COMPARE VARIABLES"
    stat_kw = [k for k, on in (("DESCRIPTIVES", statistics and descriptives),
                               ("EXTREME", outliers)) if on]
    log += f"\n  /STATISTICS {' '.join(stat_kw) or 'NONE'}"
    if m_estimators:
        log += "\n  /MESTIMATORS HUBER(1.339) ANDREW(1.34) HAMPEL(1.7,3.4,8.5) TUKEY(4.685)"
    if percentiles:
        log += "\n  /PERCENTILES(5,10,25,50,75,90,95) HAVERAGE"
    missing_kw = {"pairwise": "PAIRWISE", "report": "REPORT"}.get(missing, "LISTWISE")
    section.log(log + f"\n  /CINTERVAL {level * 100:g}\n  /MISSING {missing_kw}.")
    if not dependents:
        return section

    frame_all = pd.concat({n: numeric(ds, n) for n in dependents} |
                          {f: ds.analysis_series(f) for f in factors}, axis=1)
    if label_by:
        frame_all["__label"] = ds.analysis_series(label_by)
    frame_all["__case"] = np.arange(1, len(frame_all) + 1)
    total = len(frame_all)
    if missing == "report":
        for f in factors:
            frame_all[f] = frame_all[f].astype(object).where(frame_all[f].notna(), "__missing")
    if missing == "listwise":
        frame = frame_all.dropna(subset=dependents + factors)
    else:
        frame = frame_all.dropna(subset=factors) if factors else frame_all

    splits = [(f, ds.var(f)) for f in factors] or [(None, None)]
    has_factor = bool(factors)
    rh = 2 if has_factor else 1

    def groups(dep, factor):
        sub = frame.dropna(subset=[dep])
        return _groups(ds, sub, dep, factor)

    columns = [""] * rh + [("Valid", "N"), ("Valid", "Percent"), ("Missing", "N"),
                           ("Missing", "Percent"), ("Total", "N"), ("Total", "Percent")]
    rows = []
    for dep in dependents:
        title = ds.var(dep).title
        for factor, fvar in splits:
            for i, g in enumerate(groups(dep, factor)):
                if has_factor:
                    fac = frame_all[factor]
                    n_total = int((fac == g.value).sum())
                else:
                    n_total = total
                n_valid = g.n
                head = [title if i == 0 else "", f"{fvar.title} = {g.label}"] if has_factor \
                    else [title]
                rows.append(head + [str(n_valid), _pct(n_valid, n_total),
                                    str(n_total - n_valid), _pct(n_total - n_valid, n_total),
                                    str(n_total), "100.0%" if n_total else ""])
    section.table(Table("Case Processing Summary", columns, rows, row_header_cols=rh))

    def each():
        for dep in dependents:
            var = ds.var(dep)
            for factor, fvar in splits:
                for gi, g in enumerate(groups(dep, factor)):
                    yield dep, var, factor, fvar, gi, g

    if statistics and descriptives:
        drows = []
        for dep, var, factor, fvar, gi, g in each():
            for si, (name, stat, se) in enumerate(_descriptive_rows(g.data, var.decimals,
                                                                      level)):
                first = gi == 0 and si == 0
                head = [var.title if first else ""]
                if has_factor:
                    head.append(f"{fvar.title} = {g.label}" if si == 0 else "")
                drows.append(head + [name, stat, se])
        section.table(Table("Descriptives", [""] * (rh + 1) + ["Statistic", "Std. Error"],
                            drows, row_header_cols=rh + 1))

    if statistics and m_estimators:
        mrows = []
        for dep, var, factor, fvar, gi, g in each():
            head = [var.title if gi == 0 else ""] + (
                [f"{fvar.title} = {g.label}"] if has_factor else [])
            mrows.append(head + [fmt(m_estimator(g.data, k), var.decimals + 4)
                                 for k in ("huber", "tukey", "hampel", "andrews")])
        section.table(Table("M-Estimators", [""] * rh + [
            "Huber's M-Estimatorᵃ", "Tukey's Biweightᵇ", "Hampel's M-Estimatorᶜ",
            "Andrews' Waveᵈ"], mrows, row_header_cols=rh, footnotes=[
            "a. The weighting constant is 1.339.", "b. The weighting constant is 4.685.",
            "c. The weighting constants are 1.700, 3.400, and 8.500",
            "d. The weighting constant is 1.340*pi."]))

    if statistics and percentiles:
        prows = []
        for dep, var, factor, fvar, gi, g in each():
            if g.n == 0:
                continue
            wa = weighted_average_percentiles(g.data)
            th = tukey_hinges(g.data)
            head = [var.title if gi == 0 else ""] + (
                [f"{fvar.title} = {g.label}"] if has_factor else [])
            d = var.decimals + 4
            prows.append(["Weighted Average(Definition 1)" if gi == 0 and dep == dependents[0]
                          else ""] + head + [fmt(v, d) for v in wa])
            prows.append(["Tukey's Hinges" if gi == 0 and dep == dependents[0] else ""] +
                         [""] * rh + ["", "", fmt(th[0], d), fmt(th[1], d), fmt(th[2], d), "",
                                      ""])
        section.table(Table("Percentiles", [""] * (rh + 1) + [
            ("Percentiles", f"{p}") for p in PERCENTILES], prows, row_header_cols=rh + 1))

    if statistics and outliers:
        erows, notes = [], []
        label_var = ds.var(label_by) if label_by else None
        for dep, var, factor, fvar, gi, g in each():
            sub = frame.dropna(subset=[dep])
            if factor is not None:
                sub = sub[sub[factor] == g.value]
            x = sub[dep].to_numpy(float)
            cases = sub["__case"].to_numpy()
            if len(x) == 0:
                continue
            ev = extreme_values(x, cases)
            for side, title_side in (("high", "Highest"), ("low", "Lowest")):
                for rank, (case, value, idx) in enumerate(ev[side], start=1):
                    first = rank == 1
                    head = [var.title if gi == 0 and side == "high" and first else ""]
                    if has_factor:
                        head.append(f"{fvar.title} = {g.label}" if side == "high" and first
                                    else "")
                    mark = ""
                    if rank == len(ev[side]) and ev[side + "_partial"] is not None:
                        letter = chr(ord("a") + len(notes))
                        mark = letter
                        notes.append(f"{letter}. Only a partial list of cases with the value "
                                     f"{_short(value)} are shown in the table of "
                                     f"{'upper' if side == 'high' else 'lower'} extremes.")
                    row = head + [title_side if first else "", str(rank), str(case)]
                    if label_var is not None:
                        row.append(value_text(label_var, sub["__label"].iloc[idx])
                                   if pd.notna(sub["__label"].iloc[idx]) else "")
                    row.append(fmt(value, var.decimals) + (_sup(mark) if mark else ""))
                    erows.append(row)
        cols = [""] * (rh + 2) + ["Case Number"] + ([label_var.title] if label_var else []) \
            + ["Value"]
        section.table(Table("Extreme Values", cols, erows, row_header_cols=rh + 2,
                            footnotes=notes))

    if normality:
        nrows, lower_bound = [], False
        for dep, var, factor, fvar, gi, g in each():
            x = g.data
            d, p, lb = lilliefors(x)
            w, sp = shapiro(x)
            lower_bound |= lb
            head = [var.title if gi == 0 else ""]
            if has_factor:
                head.append(f"{fvar.title} = {g.label}")
            nrows.append(head + [fmt(d), fmt_int(len(x)), fmt_p(p) + ("*" if lb else ""),
                                 fmt(w), fmt_int(len(x)), fmt_p(sp)])
        notes = []
        if lower_bound:
            notes.append("*. This is a lower bound of the true significance.")
        notes.append("a. Lilliefors Significance Correction")
        section.table(Table("Tests of Normality", [""] * rh + [
            ("Kolmogorov-Smirnovᵃ", "Statistic"), ("Kolmogorov-Smirnovᵃ", "df"),
            ("Kolmogorov-Smirnovᵃ", "Sig."), ("Shapiro-Wilk", "Statistic"),
            ("Shapiro-Wilk", "df"), ("Shapiro-Wilk", "Sig.")], nrows, row_header_cols=rh,
            footnotes=notes))

    if spread != "none" and has_factor:
        _spread_level(section, ds, dependents, splits, groups, spread, spread_power)

    for dep in dependents:
        var = ds.var(dep)
        for factor, fvar in splits:
            gs = groups(dep, factor)
            for g in gs:
                x = g.data
                suffix = f" ({fvar.title} = {g.label})" if factor else ""
                if histogram and len(x):
                    section.image(f"Histogram of {var.title}{suffix}",
                                  charts.histogram(x, f"Histogram{suffix}", var.title, False))
                if stem_leaf and len(x):
                    section.table(_stem_table(x, f"{var.title} Stem-and-Leaf Plot{suffix}"))
                if qq and len(x) > 2:
                    section.image(f"Normal Q-Q Plot of {var.title}{suffix}",
                                  charts.qq_plot(x, f"Normal Q-Q Plot of {var.title}{suffix}"))
                    section.image(f"Detrended Normal Q-Q Plot of {var.title}{suffix}",
                                  detrended_qq_plot(x, f"Detrended Normal Q-Q Plot of "
                                                       f"{var.title}{suffix}"))
            if boxplot_mode == "factor" and gs:
                section.image(f"Boxplot of {var.title}" + (f" by {fvar.title}" if factor else ""),
                              charts.boxplot([g.data for g in gs],
                                             [g.label for g in gs] if factor
                                             else [var.title], var.title, var.title))
    if boxplot_mode == "dependents":
        for factor, fvar in splits:
            if factor is None:
                data = [groups(d, None)[0].data for d in dependents]
                section.image("Boxplot", charts.boxplot(
                    data, [ds.var(d).title for d in dependents], "Boxplot", ""))
            else:
                levels = groups(dependents[0], factor)
                for lv in levels:
                    data = []
                    for d in dependents:
                        match = [g for g in groups(d, factor) if g.value == lv.value]
                        data.append(match[0].data if match else np.array([]))
                    section.image(f"Boxplot ({fvar.title} = {lv.label})", charts.boxplot(
                        data, [ds.var(d).title for d in dependents],
                        f"{fvar.title} = {lv.label}", ""))
    return section


def _sup(letter: str) -> str:
    from rsstatchem.core.stats.common import sup
    return sup(letter)


def _spread_level(section, ds, dependents, splits, groups, spread, spread_power) -> None:
    rows = []
    for dep in dependents:
        var = ds.var(dep)
        for factor, fvar in splits:
            gs = groups(dep, factor)
            power = None
            if spread == "transformed":
                power = SPREAD_POWERS.get(spread_power, 0.0)
            samples = [g.data if power is None else _transform(g.data, power) for g in gs]
            for i, (title, f, df1, df2, p) in enumerate(levene_examine(samples)):
                rows.append([var.title if i == 0 else "", title, fmt(f), fmt_int(df1),
                             fmt_int(df2) if float(df2).is_integer() else fmt(df2), fmt_p(p)])
            sl = spread_level([g.data for g in gs], power)
            if len(sl["x"]) >= 2:
                head = "Spread vs. Level Plot of " + var.title + f" by {fvar.title}"
                note = (f"Data transformed using power = {power:g} ({_POWER_TEXT.get(power, '')})"
                        if power is not None else
                        f"Slope = {sl['slope']:.3f}   Power for transformation = "
                        f"{sl['power']:.3f}")
                section.image(head, spread_level_plot(sl, head, note))
    title = "Test of Homogeneity of Variance"
    notes = []
    if spread == "transformed":
        p = SPREAD_POWERS.get(spread_power, 0.0)
        notes.append(f"Levene's test computed on data transformed by power {p:g} "
                     f"({_POWER_TEXT.get(p, '')}).")
    section.table(Table(title, ["", "", "Levene Statistic", "df1", "df2", "Sig."], rows,
                        row_header_cols=2, footnotes=notes))


def _stem_table(x: np.ndarray, title: str) -> Table:
    sl = stem_and_leaf(x)
    rows = [[f"{freq:.2f}", stem, leaves] for freq, stem, leaves in sl["lines"]]
    width_text = f"{sl['width']:g}"
    return Table(title, ["Frequency", "Stem &", "Leaf"], rows, row_header_cols=0,
                 footnotes=[f"Stem width: {width_text}",
                            f"Each leaf: {sl['per_leaf']} case(s)"])


def detrended_qq_plot(values: np.ndarray, title: str) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    x = np.sort(np.asarray(values, float))
    n = len(x)
    fig = Figure(figsize=(5.2, 4.2), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    mean, sd = x.mean(), x.std(ddof=1)
    expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    dev = (x - mean) / sd - expected if sd > 0 else np.zeros(n)
    ax.plot(x, dev, "o", color=charts.BAR, markeredgecolor=charts.EDGE, markersize=5)
    ax.axhline(0, color=charts.LINE, lw=1.2)
    charts._style(ax, title, "Observed Value", "Dev from Normal")
    return charts._png(fig)


def spread_level_plot(sl: dict, title: str, note: str) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 4.2), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.plot(sl["x"], sl["y"], "o", color=charts.BAR, markeredgecolor=charts.EDGE)
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8)
    charts._style(ax, title, "Level (ln of Median)", "Spread (ln of IQR)")
    return charts._png(fig)


def _descriptive_rows(x: np.ndarray, decimals: int, level: float) -> list[tuple[str, str, str]]:
    r = summary(pd.Series(x))
    if r["n"] == 0:
        return []
    lo, hi = t_interval(r["mean"], r["sem"], r["n"] - 1, level)
    q1, q3 = percentile(x, [25, 75])
    d = decimals
    return [
        ("Mean", fmt(r["mean"], d + 2), fmt(r["sem"], d + 3)),
        (f"{pct(level)} Confidence Interval for Mean: Lower Bound", fmt(lo, d + 2), ""),
        (f"{pct(level)} Confidence Interval for Mean: Upper Bound", fmt(hi, d + 2), ""),
        ("5% Trimmed Mean", fmt(trimmed_mean(x), d + 2), ""),
        ("Median", fmt(r["median"], d + 2), ""),
        ("Variance", fmt(r["var"], d + 3), ""),
        ("Std. Deviation", fmt(r["std"], d + 3), ""),
        ("Minimum", fmt(r["min"], d), ""),
        ("Maximum", fmt(r["max"], d), ""),
        ("Range", fmt(r["range"], d), ""),
        ("Interquartile Range", fmt(q3 - q1, d + 2), ""),
        ("Skewness", fmt(r["skew"]), fmt(r["se_skew"])),
        ("Kurtosis", fmt(r["kurt"]), fmt(r["se_kurt"])),
    ]


def _groups(ds: Dataset, frame: pd.DataFrame, dep: str, factor: str | None) -> list[Group]:
    if factor is None:
        return [Group(None, "", frame[dep].to_numpy(float))]
    sub = pd.DataFrame({"y": frame[dep], "g": frame[factor]}).dropna()
    if (sub["g"] == "__missing").any():
        out = groups_of(ds, dep, factor, frame=sub[sub["g"] != "__missing"])
        miss = sub.loc[sub["g"] == "__missing", "y"].to_numpy(float)
        return out + [Group("__missing", "Missing", miss)]
    return groups_of(ds, dep, factor, frame=sub)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else ""


__all__ = ["explore", "m_estimator", "tukey_hinges", "extreme_values", "levene_examine",
           "stem_and_leaf", "spread_level"]

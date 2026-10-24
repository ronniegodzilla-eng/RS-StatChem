"""Analyze > Mediation & Moderation: regression-based mediation and moderation in the layout of
Hayes' PROCESS macro (model numbers follow Hayes, 2022).

Supported: Model 4 (simple and parallel multiple mediation), Model 1 (simple moderation with
optional mean centring, conditional effects at the 16th/50th/84th percentiles or mean ± 1 SD,
Johnson-Neyman region, interaction plot), Model 7 (first-stage moderated mediation) and Model
14 (second-stage moderated mediation). OLS with optional HC0-HC4 standard errors; indirect
effects with percentile bootstrap confidence intervals (user-set number of samples and seed),
completely standardised indirect effects, pairwise contrasts and the Sobel test.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, listwise, percentile

MODELS = {1: "Simple moderation", 4: "Mediation (simple / parallel)",
          7: "Moderated mediation (first stage)", 14: "Moderated mediation (second stage)"}
HC_TYPES = (None, "HC0", "HC1", "HC2", "HC3", "HC4")
D = 4  # PROCESS prints four decimals


def f4(x) -> str:
    return fmt(x, D)


@dataclass
class ProcessResult:
    section: Section
    effects: dict = field(default_factory=dict)  # numeric results for tests / reuse


# ------------------------------------------------------------------ OLS
@dataclass
class OLS:
    names: list[str]
    b: np.ndarray
    cov: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    df: int
    r2: float
    mse: float
    F: float
    df1: int
    p_model: float
    n: int


def ols(y: np.ndarray, X: np.ndarray, names: list[str], hc: str | None = None) -> OLS:
    """X includes the constant in column 0."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    df = n - k
    sse = float(e @ e)
    mse = sse / df
    if hc is None:
        cov = XtX_inv * mse
    else:
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        e2 = e ** 2
        if hc == "HC0":
            w = e2
        elif hc == "HC1":
            w = e2 * n / df
        elif hc == "HC2":
            w = e2 / (1 - h)
        elif hc == "HC3":
            w = e2 / (1 - h) ** 2
        else:
            delta = np.minimum(4.0, n * h / k)
            w = e2 / (1 - h) ** delta
        cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), df)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    df1 = k - 1
    if df1 > 0:
        if hc is None:
            F = (r2 / df1) / ((1 - r2) / df)
        else:
            bb = b[1:]
            F = float(bb @ np.linalg.solve(cov[1:, 1:], bb)) / df1
        p_model = float(stats.f.sf(F, df1, df))
    else:
        F, p_model = float("nan"), float("nan")
    return OLS(names, b, cov, se, t, p, df, r2, mse, float(F), df1, p_model, n)


def _boot_coefs(designs: list[tuple[np.ndarray, np.ndarray]], idx: np.ndarray) -> list:
    """Coefficients of each (y, X) equation for a block of bootstrap resamples (B x n idx)."""
    out = []
    for y, X in designs:
        Xb = X[idx]
        yb = y[idx]
        XtX = np.einsum("bni,bnj->bij", Xb, Xb)
        Xty = np.einsum("bni,bn->bi", Xb, yb)
        out.append(np.einsum("bij,bj->bi", np.linalg.pinv(XtX), Xty))
    return out


def boot_ci(values: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval as PROCESS computes it (ordered bootstrap estimates)."""
    v = np.sort(values[np.isfinite(values)])
    B = len(v)
    if B == 0:
        return float("nan"), float("nan")
    low = 0.5 * (1 - level)
    lo = max(int(math.floor(B * low)), 1)
    hi = min(int(math.floor(B * (1 - low))) + 1, B)
    return float(v[lo - 1]), float(v[hi - 1])


# ------------------------------------------------------------------ tables
def _summary_table(title: str, r: OLS, hc: str | None) -> Table:
    return Table(title, ["R", "R-sq", "MSE", "HC-F" if hc else "F", "df1", "df2", "p"],
                 [[f4(math.sqrt(max(r.r2, 0))), f4(r.r2), f4(r.mse), f4(r.F), fmt_int(r.df1),
                   fmt_int(r.df), f4(r.p_model)]], row_header_cols=0)


def _coef_table(title: str, r: OLS, level: float, hc: str | None, labels: list[str]) -> Table:
    q = stats.t.ppf(0.5 + level / 2, r.df)
    rows = [[lab, f4(b), f4(se), f4(t), f4(p), f4(b - q * se), f4(b + q * se)]
            for lab, b, se, t, p in zip(labels, r.b, r.se, r.t, r.p)]
    return Table(title, ["", "coeff", f"se({hc})" if hc else "se", "t", "p", "LLCI", "ULCI"],
                 rows)


def _boot_cols(prefix: str = "Effect") -> list[str]:
    return [prefix, "BootSE", "BootLLCI", "BootULCI"]


def _cond_values(w: np.ndarray, moments: bool) -> tuple[list[float], str]:
    uniq = np.unique(w)
    if len(uniq) == 2:
        return [float(u) for u in uniq], "the two values of the dichotomous moderator"
    if moments:
        m, s = float(w.mean()), float(w.std(ddof=1))
        return [m - s, m, m + s], "the mean and plus/minus one SD from the mean"
    return [float(v) for v in percentile(w, [16, 50, 84])], \
        "the 16th, 50th, and 84th percentiles"


def _jn(b1, b3, v11, v33, v13, df, level, wmin, wmax) -> list[float]:
    """Moderator values where the conditional effect b1 + b3 W is exactly significant."""
    t2 = stats.t.ppf(0.5 + level / 2, df) ** 2
    a = b3 * b3 - t2 * v33
    b = 2 * (b1 * b3 - t2 * v13)
    c = b1 * b1 - t2 * v11
    if abs(a) < 1e-14:
        roots = [-c / b] if b else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        roots = [(-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)]
    return sorted(r for r in roots if wmin <= r <= wmax)


def _interaction_plot(xv, wv, yhat, xname, wname, yname) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.0, 4.2), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    colors = ["#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#EF6C00"]
    styles = ["--", "-", ":", "-.", "-"]
    for i, w in enumerate(wv):
        ax.plot(xv, yhat[i], styles[i % 5], color=colors[i % 5], marker="o", linewidth=1.7,
                label=f"{wname} = {w:.3f}")
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(f"Conditional effect of {xname} on {yname}", fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


# ------------------------------------------------------------------ main
def process(ds: Dataset, model: int, y: str, x: str, mediators: list[str] | None = None,
            w: str | None = None, covariates: list[str] | None = None, boot: int = 5000,
            seed: int | None = None, level: float = 0.95, center: bool = False,
            moments: bool = False, jn: bool = True, total: bool = False,
            effsize: bool = True, contrast: bool = False, normal: bool = False,
            hc: str | None = None, plot: bool = True) -> ProcessResult:
    mediators = mediators or []
    covariates = covariates or []
    section = Section(f"PROCESS Model {model}")
    result = ProcessResult(section)
    if model not in MODELS:
        section.note("Warning", f"Model {model} is not available.", style="warning")
        return result
    if model in (4, 7, 14) and not mediators:
        section.note("Warning", "Specify at least one mediator (M).", style="warning")
        return result
    if model in (1, 7, 14) and not w:
        section.note("Warning", "Specify a moderator (W).", style="warning")
        return result
    if model == 1:
        mediators = []
    names = [y, x] + mediators + ([w] if w and model != 4 else []) + covariates
    if len(set(names)) != len(names):
        section.note("Warning", "Each variable may appear in only one role.", style="warning")
        return result
    data = listwise(ds, names)
    n = len(data)
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2 ** 31 - 1))
    vn = {k: ds.var(k).name for k in names}
    log = (f"PROCESS y={vn[y]} /x={vn[x]}" + (f" /m={' '.join(vn[m] for m in mediators)}"
                                              if mediators else "") +
           (f" /w={vn[w]}" if w and model != 4 else "") +
           (f" /cov={' '.join(vn[c] for c in covariates)}" if covariates else "") +
           f" /model={model}" + (f" /boot={boot} /seed={seed}" if model != 1 else "") +
           f" /conf={level * 100:g}" +
           (" /center=1" if center else "") + (" /moments=1" if moments else "") +
           (" /jn=1" if jn and model == 1 else "") + (" /total=1" if total else "") +
           (" /effsize=1" if effsize else "") + (" /contrast=1" if contrast else "") +
           (" /normal=1" if normal else "") + (f" /hc={hc[2:]}" if hc else "") + ".")
    section.log(log)
    k_needed = 3 + len(mediators) + len(covariates) + 2
    if n < k_needed + 2:
        section.note("Warning", "Not enough valid cases for the model.", style="warning")
        return result

    Yv = data[y].to_numpy(float)
    Xv = data[x].to_numpy(float)
    Ms = [data[m].to_numpy(float) for m in mediators]
    Wv = data[w].to_numpy(float) if w and model != 4 else None
    Cs = [data[c].to_numpy(float) for c in covariates]
    centered = []
    if center and model in (1, 7, 14):
        if model in (1, 7):
            Xv = Xv - Xv.mean()
            centered.append(vn[x])
        if model == 14:
            Ms = [m - m.mean() for m in Ms]
            centered += [vn[m] for m in mediators]
        Wv = Wv - Wv.mean()
        centered.append(vn[w])

    header = [["Model", str(model)], ["Y", vn[y]], ["X", vn[x]]]
    for i, m in enumerate(mediators):
        header.append([f"M{i + 1}" if len(mediators) > 1 else "M", vn[m]])
    if Wv is not None:
        header.append(["W", vn[w]])
    if covariates:
        header.append(["Covariates", " ".join(vn[c] for c in covariates)])
    header.append(["Sample Size", str(n)])
    if model != 1 and boot:
        header.append(["Custom Seed", str(seed)])
    section.table(Table(f"PROCESS-style analysis: Model {model} ({MODELS[model]})", ["", ""],
                        header, footnotes=["Model numbers follow Hayes (2022), Introduction to "
                                           "Mediation, Moderation, and Conditional Process "
                                           "Analysis."]))

    ones = np.ones(n)
    cov_cols = Cs
    cov_names = [vn[c] for c in covariates]
    designs: list[tuple[np.ndarray, np.ndarray]] = []
    eq_labels: list[list[str]] = []
    fits: list[OLS] = []
    outcome_names: list[str] = []

    def fit(yvec, cols, labels, out_name):
        X = np.column_stack([ones] + cols)
        r = ols(yvec, X, labels, hc)
        designs.append((yvec, X))
        eq_labels.append(labels)
        fits.append(r)
        outcome_names.append(out_name)
        return r

    effects = result.effects
    conf = f"{level * 100:g}"
    xw = None
    if model == 1:
        xw = Xv * Wv
        r = fit(Yv, [Xv, Wv, xw] + cov_cols, ["constant", vn[x], vn[w], "Int_1"] + cov_names,
                vn[y])
        _emit_outcome(section, vn[y], r, level, hc)
        section.note("Product terms key:", f"Int_1 : {vn[x]} x {vn[w]}")
        r0 = ols(Yv, np.column_stack([ones, Xv, Wv] + cov_cols), [], hc)
        _interaction_test(section, r, r0, 3, "X*W", hc)
        b1, b3 = r.b[1], r.b[3]
        v11, v33, v13 = r.cov[1, 1], r.cov[3, 3], r.cov[1, 3]
        q = stats.t.ppf(0.5 + level / 2, r.df)
        wvals, how = _cond_values(Wv, moments)
        rows = []
        for wv in wvals:
            eff = b1 + b3 * wv
            se = math.sqrt(max(v11 + 2 * wv * v13 + wv * wv * v33, 0))
            t = eff / se
            rows.append([f4(wv), f4(eff), f4(se), f4(t), f4(2 * stats.t.sf(abs(t), r.df)),
                         f4(eff - q * se), f4(eff + q * se)])
        section.note("", f"Focal predict: {vn[x]} (X)    Mod var: {vn[w]} (W)")
        section.table(Table("Conditional effects of the focal predictor at values of the "
                            "moderator(s):", [vn[w], "Effect", "se", "t", "p", "LLCI", "ULCI"],
                            rows, footnotes=[f"Moderator values are {how}."]))
        effects["conditional"] = [(wv, b1 + b3 * wv) for wv in wvals]
        if jn and len(np.unique(Wv)) > 2:
            pts = _jn(b1, b3, v11, v33, v13, r.df, level, Wv.min(), Wv.max())
            effects["jn"] = pts
            if pts:
                section.table(Table("Moderator value(s) defining Johnson-Neyman significance "
                                    "region(s):", ["Value", "% below", "% above"],
                                    [[f4(pv), f4(np.mean(Wv < pv) * 100),
                                      f4(np.mean(Wv > pv) * 100)] for pv in pts],
                                    row_header_cols=0))
            else:
                section.note("", "There are no statistical significance transition points "
                             "within the observed range of the moderator found using the "
                             "Johnson-Neyman method.")
            grid = list(np.linspace(Wv.min(), Wv.max(), 21))
            grid = sorted(grid + pts)
            rows = []
            for wv in grid:
                eff = b1 + b3 * wv
                se = math.sqrt(max(v11 + 2 * wv * v13 + wv * wv * v33, 0))
                t = eff / se
                rows.append([f4(wv), f4(eff), f4(se), f4(t), f4(2 * stats.t.sf(abs(t), r.df)),
                             f4(eff - q * se), f4(eff + q * se)])
            section.table(Table("Conditional effect of focal predictor at values of the "
                                "moderator:", [vn[w], "Effect", "se", "t", "p", "LLCI", "ULCI"],
                                rows))
        # data for visualising
        xvals, _ = _cond_values(Xv, moments)
        cmeans = [c.mean() for c in Cs]
        yhat = np.array([[r.b[0] + r.b[1] * xv + r.b[2] * wv + r.b[3] * xv * wv +
                          sum(bc * cm for bc, cm in zip(r.b[4:], cmeans)) for xv in xvals]
                         for wv in wvals])
        rows = [[f4(xv), f4(wv), f4(yhat[i, j])] for i, wv in enumerate(wvals)
                for j, xv in enumerate(xvals)]
        section.table(Table("Data for visualizing the conditional effect of the focal "
                            "predictor:", [vn[x], vn[w], vn[y]], rows, row_header_cols=0,
                            footnotes=["Covariates are set to their sample means."]
                            if Cs else []))
        if plot:
            section.image("Interaction plot", _interaction_plot(xvals, wvals, yhat, vn[x],
                                                                 vn[w], vn[y]))
        _footer(section, level, None, centered, hc, conf)
        effects["fit"] = r
        return result

    # ---------------------------------------------------------------- mediation models
    km = len(mediators)
    a_fits = []
    for i, m in enumerate(Ms):
        if model == 7:
            cols = [Xv, Wv, Xv * Wv] + cov_cols
            labels = ["constant", vn[x], vn[w], "Int_1"] + cov_names
        else:
            cols = [Xv] + cov_cols
            labels = ["constant", vn[x]] + cov_names
        r = fit(m, cols, labels, vn[mediators[i]])
        a_fits.append(r)
        _emit_outcome(section, vn[mediators[i]], r, level, hc)
        if model == 7:
            section.note("Product terms key:", f"Int_1 : {vn[x]} x {vn[w]}")
            r0 = ols(m, np.column_stack([ones, Xv, Wv] + cov_cols), [], hc)
            _interaction_test(section, r, r0, 3, "X*W", hc)
    if model == 14:
        mw = [m * Wv for m in Ms]
        cols = [Xv] + Ms + [Wv] + mw + cov_cols
        labels = ["constant", vn[x]] + [vn[m] for m in mediators] + [vn[w]] + \
            [f"Int_{i + 1}" for i in range(km)] + cov_names
    else:
        cols = [Xv] + Ms + cov_cols
        labels = ["constant", vn[x]] + [vn[m] for m in mediators] + cov_names
    ry = fit(Yv, cols, labels, vn[y])
    _emit_outcome(section, vn[y], ry, level, hc)
    if model == 14:
        section.note("Product terms key:", "; ".join(
            f"Int_{i + 1} : {vn[m]} x {vn[w]}" for i, m in enumerate(mediators)))
        r0 = ols(Yv, np.column_stack([ones, Xv] + Ms + [Wv] + cov_cols), [], hc)
        _interaction_test(section, ry, r0, list(range(3 + km, 3 + 2 * km)),
                          " ".join(f"M{i + 1 if km > 1 else ''}*W" for i in range(km)), hc)
    rt = None
    if model == 4:
        rt = ols(Yv, np.column_stack([ones, Xv] + cov_cols), ["constant", vn[x]] + cov_names,
                 hc)
        if total:
            section.note("", "************************** TOTAL EFFECT MODEL "
                         "****************************")
            _emit_outcome(section, vn[y], rt, level, hc)

    # bootstrap
    rng = np.random.default_rng(seed)
    sdx, sdy = Xv.std(ddof=1), Yv.std(ddof=1)
    wvals, how = _cond_values(Wv, moments) if model in (7, 14) else ([], "")
    a_idx = 1  # coefficient of X in the M models
    b_idx = [2 + i for i in range(km)]  # coefficients of M in the Y model

    def effect_vectors(coefs, sx, sy):
        """Rows per bootstrap sample: indirect effects (per mediator), plus extras."""
        a = np.column_stack([coefs[i][:, a_idx] for i in range(km)])
        if model == 4:
            b = np.column_stack([coefs[km][:, j] for j in b_idx])
            ind = a * b
            return {"ind": ind, "std": ind * (sx / sy)[:, None]}
        if model == 7:
            a3 = np.column_stack([coefs[i][:, 3] for i in range(km)])
            b = np.column_stack([coefs[km][:, j] for j in b_idx])
            cond = np.stack([(a + a3 * wv) * b for wv in wvals], axis=2)  # B x km x nw
            return {"cond": cond, "index": a3 * b}
        b1 = np.column_stack([coefs[km][:, j] for j in b_idx])
        b3 = np.column_stack([coefs[km][:, 3 + km + i] for i in range(km)])
        cond = np.stack([a * (b1 + b3 * wv) for wv in wvals], axis=2)
        return {"cond": cond, "index": a * b3}

    point_coefs = [f.b[None, :] for f in fits]
    point = effect_vectors(point_coefs, np.array([sdx]), np.array([sdy]))
    store: dict[str, list] = {k: [] for k in point}
    chunk = max(1, min(500, int(2e7 // max(n * 8, 1))))
    done = 0
    while done < boot:
        bsz = min(chunk, boot - done)
        idx = rng.integers(0, n, size=(bsz, n))
        coefs = _boot_coefs(designs, idx)
        sx = Xv[idx].std(axis=1, ddof=1)
        sy = Yv[idx].std(axis=1, ddof=1)
        ev = effect_vectors(coefs, sx, sy)
        for k_, v in ev.items():
            store[k_].append(v)
        done += bsz
    bs = {k_: np.concatenate(v, axis=0) for k_, v in store.items()} if boot else {}

    def brow(label, est, samples):
        lo, hi = boot_ci(samples, level) if boot else (float("nan"), float("nan"))
        sd = float(np.std(samples, ddof=1)) if boot else float("nan")
        return [label, f4(est), f4(sd), f4(lo), f4(hi)]

    section.note("", "************** TOTAL, DIRECT, AND INDIRECT EFFECTS OF X ON Y "
                 "**************" if model == 4 else
                 "****************** DIRECT AND INDIRECT EFFECTS OF X ON Y "
                 "*****************")
    q = stats.t.ppf(0.5 + level / 2, ry.df)
    cs = sdx / sdy
    if model == 4:
        c, sec_ = rt.b[1], rt.se[1]
        qt = stats.t.ppf(0.5 + level / 2, rt.df)
        cols = ["Effect", "se", "t", "p", "LLCI", "ULCI"] + (["c_cs"] if effsize else [])
        row = [f4(c), f4(sec_), f4(rt.t[1]), f4(rt.p[1]), f4(c - qt * sec_), f4(c + qt * sec_)]
        section.table(Table("Total effect of X on Y:", cols,
                            [row + ([f4(c * cs)] if effsize else [])], row_header_cols=0))
        effects["total"] = c
    cprime = ry.b[1]
    cols = ["Effect", "se", "t", "p", "LLCI", "ULCI"] + \
        (["c'_cs"] if effsize and model == 4 else [])
    row = [f4(cprime), f4(ry.se[1]), f4(ry.t[1]), f4(ry.p[1]), f4(cprime - q * ry.se[1]),
           f4(cprime + q * ry.se[1])]
    section.table(Table("Direct effect of X on Y:", cols,
                        [row + ([f4(cprime * cs)] if effsize and model == 4 else [])],
                        row_header_cols=0))
    effects["direct"] = cprime

    if model == 4:
        ind = point["ind"][0]
        rows = []
        if km > 1:
            rows.append(brow("TOTAL", ind.sum(), bs["ind"].sum(axis=1) if boot else None))
        for i in range(km):
            rows.append(brow(vn[mediators[i]], ind[i], bs["ind"][:, i] if boot else None))
        con_rows = []
        defs = []
        if contrast and km > 1:
            ci = 1
            for i in range(km):
                for j in range(i + 1, km):
                    con_rows.append(brow(f"(C{ci})", ind[i] - ind[j],
                                         bs["ind"][:, i] - bs["ind"][:, j] if boot else None))
                    defs.append([f"(C{ci})", vn[mediators[i]], "minus", vn[mediators[j]]])
                    ci += 1
        section.table(Table("Indirect effect(s) of X on Y:", [""] + _boot_cols(),
                            rows + con_rows))
        if defs:
            section.table(Table("Specific indirect effect contrast definition(s):",
                                ["", "", "", ""], defs))
        effects["indirect"] = ind.tolist()
        if boot:
            effects["indirect_ci"] = [boot_ci(bs["ind"][:, i], level) for i in range(km)]
        if effsize:
            sind = point["std"][0]
            rows = []
            if km > 1:
                rows.append(brow("TOTAL", sind.sum(), bs["std"].sum(axis=1) if boot else None))
            for i in range(km):
                rows.append(brow(vn[mediators[i]], sind[i], bs["std"][:, i] if boot else None))
            section.table(Table("Completely standardized indirect effect(s) of X on Y:",
                                [""] + _boot_cols(), rows))
            effects["std_indirect"] = sind.tolist()
        if normal:
            rows = []
            for i in range(km):
                a, sa = a_fits[i].b[1], a_fits[i].se[1]
                b, sb = ry.b[2 + i], ry.se[2 + i]
                se = math.sqrt(a * a * sb * sb + b * b * sa * sa)
                z = a * b / se
                rows.append([vn[mediators[i]], f4(a * b), f4(se), f4(z),
                             f4(2 * stats.norm.sf(abs(z)))])
            section.table(Table("Normal theory test for indirect effect(s):",
                                ["", "Effect", "se", "Z", "p"], rows))
    else:
        cond = point["cond"][0]
        for i in range(km):
            rows = [brow(f4(wv), cond[i, j], bs["cond"][:, i, j] if boot else None)
                    for j, wv in enumerate(wvals)]
            section.table(Table(f"Conditional indirect effects of X on Y — INDIRECT EFFECT: "
                                f"{vn[x]} -> {vn[mediators[i]]} -> {vn[y]}",
                                [vn[w]] + _boot_cols(), rows,
                                footnotes=[f"Moderator values are {how}."]))
            idx_row = brow(vn[w], point["index"][0, i], bs["index"][:, i] if boot else None)
            section.table(Table(f"Index of moderated mediation — {vn[mediators[i]]}:",
                                ["", "Index", "BootSE", "BootLLCI", "BootULCI"], [idx_row]))
        effects["conditional_indirect"] = cond.tolist()
        effects["index"] = point["index"][0].tolist()
        if boot:
            effects["index_ci"] = [boot_ci(bs["index"][:, i], level) for i in range(km)]
    effects["fits"] = fits
    _footer(section, level, boot, centered, hc, conf)
    return result


def _emit_outcome(section: Section, name: str, r: OLS, level: float, hc) -> None:
    section.table(_summary_table(f"OUTCOME VARIABLE: {name} — Model Summary", r, hc))
    section.table(_coef_table(f"OUTCOME VARIABLE: {name} — Model", r, level, hc, r.names))


def _interaction_test(section: Section, full: OLS, reduced: OLS, cols, label: str, hc) -> None:
    cols = [cols] if isinstance(cols, int) else list(cols)
    df1 = len(cols)
    if hc is None:
        dr2 = full.r2 - reduced.r2
        F = (dr2 / df1) / ((1 - full.r2) / full.df)
    else:
        bb = full.b[cols]
        F = float(bb @ np.linalg.solve(full.cov[np.ix_(cols, cols)], bb)) / df1
        dr2 = full.r2 - reduced.r2
    p = stats.f.sf(F, df1, full.df)
    section.table(Table("Test(s) of highest order unconditional interaction(s):",
                        ["", "R2-chng", "F", "df1", "df2", "p"],
                        [[label, f4(dr2), f4(F), fmt_int(df1), fmt_int(full.df), f4(p)]]))


def _footer(section: Section, level, boot, centered, hc, conf) -> None:
    rows = [["Level of confidence for all confidence intervals in output", f"{level * 100:.4f}"]]
    if boot:
        rows.append(["Number of bootstrap samples for percentile bootstrap confidence intervals",
                     str(boot)])
    if centered:
        rows.append(["Variables mean centered prior to analysis", " ".join(centered)])
    if hc:
        rows.append(["Heteroscedasticity consistent standard error and covariance matrix "
                     "estimator", hc])
    section.table(Table("ANALYSIS NOTES AND ERRORS", ["", ""], rows))


def selftest_data(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    m = 0.5 * x + rng.normal(0, 1, n)
    y = 0.4 * m + 0.2 * x + 0.3 * x * w + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "m": m, "w": w, "y": y})


__all__ = ["process", "ols", "boot_ci", "MODELS", "HC_TYPES"]

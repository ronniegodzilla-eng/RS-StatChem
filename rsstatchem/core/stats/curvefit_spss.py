"""Analyze > Regression > Curve Estimation (SPSS CURVEFIT).

Eleven models fitted by ordinary least squares on the linearised scale, exactly as SPSS does,
so R Square, F and Sig. refer to the transformed equation (e.g. ln Y for Compound):

    Linear       Y = b0 + b1*t              Compound     Y = b0 * b1**t
    Logarithmic  Y = b0 + b1*ln(t)          Power        Y = b0 * t**b1
    Inverse      Y = b0 + b1/t              S            Y = exp(b0 + b1/t)
    Quadratic    Y = b0 + b1*t + b2*t**2    Growth       Y = exp(b0 + b1*t)
    Cubic        ... + b3*t**3              Exponential  Y = b0 * exp(b1*t)
    Logistic     Y = 1 / (1/u + b0 * b1**t)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric
from rsstatchem.core.stats.roc import PALETTE, figure, png, style

MODELS = {"linear": "Linear", "logarithmic": "Logarithmic", "inverse": "Inverse",
          "quadratic": "Quadratic", "cubic": "Cubic", "compound": "Compound",
          "power": "Power", "s": "S", "growth": "Growth", "exponential": "Exponential",
          "logistic": "Logistic"}
KEYWORDS = {"linear": "LINEAR", "logarithmic": "LOGARITHMIC", "inverse": "INVERSE",
            "quadratic": "QUADRATIC", "cubic": "CUBIC", "compound": "COMPOUND",
            "power": "POWER", "s": "S", "growth": "GROWTH", "exponential": "EXPONENTIAL",
            "logistic": "LGSTIC"}


def _coef(v: float) -> str:
    """SPSS switches to scientific notation for tiny coefficients (e.g. -4.564E-5)."""
    if v != 0 and abs(v) < 0.0005:
        mant, exp_ = f"{v:.3E}".split("E")
        return f"{mant}E{int(exp_)}"
    return fmt(v)


def _transform(model: str, t: np.ndarray, y: np.ndarray, upper: float | None):
    """Linearised (design columns, response, mask of usable cases) for one model."""
    ok = np.ones(len(t), bool)
    if model in ("logarithmic", "power"):
        ok &= t > 0
    if model in ("inverse", "s"):
        ok &= t != 0
    if model in ("compound", "power", "s", "growth", "exponential", "logistic"):
        ok &= y > 0
    if model == "logistic" and upper is not None:
        ok &= y < upper
    tt = np.where(ok, t, 1.0)
    yy = np.where(ok, y, 1.0)
    cols = {"linear": [tt], "logarithmic": [np.log(np.abs(tt))], "inverse": [1 / tt],
            "quadratic": [tt, tt ** 2], "cubic": [tt, tt ** 2, tt ** 3], "compound": [tt],
            "power": [np.log(np.abs(tt))], "s": [1 / tt], "growth": [tt],
            "exponential": [tt], "logistic": [tt]}[model]
    if model in ("linear", "logarithmic", "inverse", "quadratic", "cubic"):
        resp = yy
    elif model == "logistic":
        inv_u = 0.0 if upper is None else 1 / upper
        with np.errstate(divide="ignore", invalid="ignore"):
            resp = np.log(1 / yy - inv_u)
    else:
        resp = np.log(yy)
    return np.column_stack(cols), resp, ok


def fit_model(model: str, t: np.ndarray, y: np.ndarray, constant: bool = True,
              upper: float | None = None) -> dict:
    """R², F, df and the parameters on SPSS's reported scale (b0 = Constant)."""
    X, resp, ok = _transform(model, np.asarray(t, float), np.asarray(y, float), upper)
    X, resp = X[ok], resp[ok]
    n, k = X.shape
    design = np.column_stack([np.ones(n), X]) if constant else X
    if n <= design.shape[1]:
        return {"ok": False, "n": n}
    coef, *_ = np.linalg.lstsq(design, resp, rcond=None)
    fitted = design @ coef
    sse = float(np.sum((resp - fitted) ** 2))
    sst = float(np.sum((resp - resp.mean()) ** 2)) if constant else float(np.sum(resp ** 2))
    df1, df2 = k, n - design.shape[1]
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    f = ((sst - sse) / df1) / (sse / df2) if sse > 0 and df2 > 0 else float("inf")
    p = float(stats.f.sf(f, df1, df2)) if math.isfinite(f) else 0.0
    b0 = coef[0] if constant else (0.0 if model in ("linear", "logarithmic", "inverse",
                                                    "quadratic", "cubic", "s", "growth")
                                   else 1.0)
    slopes = list(coef[1:] if constant else coef)
    if model in ("compound", "power", "exponential", "logistic"):
        b0 = math.exp(b0) if constant else 1.0
    if model in ("compound", "logistic"):
        slopes = [math.exp(slopes[0])]
    return {"ok": True, "n": n, "r2": r2, "f": f, "df1": df1, "df2": df2, "p": p,
            "b0": b0, "b": slopes, "constant": constant}


def predict(model: str, res: dict, t: np.ndarray, upper: float | None = None) -> np.ndarray:
    t = np.asarray(t, float)
    b0, b = res["b0"], res["b"]
    with np.errstate(all="ignore"):
        if model == "linear":
            return b0 + b[0] * t
        if model == "logarithmic":
            return b0 + b[0] * np.log(t)
        if model == "inverse":
            return b0 + b[0] / t
        if model == "quadratic":
            return b0 + b[0] * t + b[1] * t ** 2
        if model == "cubic":
            return b0 + b[0] * t + b[1] * t ** 2 + b[2] * t ** 3
        if model == "compound":
            return b0 * b[0] ** t
        if model == "power":
            return b0 * t ** b[0]
        if model == "s":
            return np.exp(b0 + b[0] / t)
        if model == "growth":
            return np.exp(b0 + b[0] * t)
        if model == "exponential":
            return b0 * np.exp(b[0] * t)
        if model == "logistic":
            inv_u = 0.0 if upper is None else 1 / upper
            return 1 / (inv_u + b0 * b[0] ** t)
    raise ValueError(model)


def curve_estimation(ds: Dataset, dependents: list[str], independent: str | None,
                     models=("linear",), constant: bool = True, plot: bool = True,
                     upper: float | None = None, anova: bool = False) -> Section:
    """``independent`` None uses Time (the case sequence 1..n) as in SPSS."""
    models = [m for m in MODELS if m in models]
    section = Section("Curve Fit")
    section.log("TSET NEWVAR=NONE.\nCURVEFIT\n  /VARIABLES=" + " ".join(dependents)
                + (f" WITH {independent}" if independent else "")
                + ("\n  /CONSTANT" if constant else "\n  /NOCONSTANT")
                + "\n  /MODEL=" + " ".join(KEYWORDS[m] for m in models)
                + (f"\n  /UPPERBOUND={upper:g}" if "logistic" in models and upper else "")
                + ("\n  /PRINT ANOVA" if anova else "")
                + ("\n  /PLOT FIT" if plot else "\n  /PLOT NONE") + ".")
    ivar_title = ds.var(independent).title if independent else "Sequence"
    for dep in dependents:
        dvar = ds.var(dep)
        frame = pd.DataFrame({"y": numeric(ds, dep)})
        frame["t"] = numeric(ds, independent) if independent else \
            np.arange(1, ds.n_cases + 1, dtype=float)
        frame = frame.dropna()
        t, y = frame["t"].to_numpy(float), frame["y"].to_numpy(float)
        rows, fits = [], []
        maxb = max([{"quadratic": 2, "cubic": 3}.get(m, 1) for m in models] or [1])
        for m in models:
            res = fit_model(m, t, y, constant, upper)
            if not res["ok"]:
                rows.append([MODELS[m]] + [""] * (6 + maxb))
                continue
            fits.append((m, res))
            bs = [_coef(v) for v in res["b"]] + [""] * (maxb - len(res["b"]))
            rows.append([MODELS[m], fmt(res["r2"]), fmt(res["f"]), fmt_int(res["df1"]),
                         fmt_int(res["df2"]), fmt_p(res["p"]),
                         _coef(res["b0"]) if constant or m in ("compound", "power",
                                                            "exponential", "logistic") else ""]
                        + bs)
            if anova:
                _anova_table(section, dvar.title, MODELS[m], m, res, t, y, upper)
        cols = ["Equation"] + [("Model Summary", c) for c in ("R Square", "F", "df1", "df2",
                                                              "Sig.")] \
            + [("Parameter Estimates", "Constant")] \
            + [("Parameter Estimates", f"b{i}") for i in range(1, maxb + 1)]
        notes = [f"Dependent Variable: {dvar.title}",
                 f"The independent variable is {ivar_title}."]
        if any(m in models for m in ("compound", "power", "s", "growth", "exponential",
                                     "logistic")):
            dropped = [m for m in models if m in ("compound", "power", "s", "growth",
                                                  "exponential", "logistic")
                       and fit_model(m, t, y, constant, upper).get("n", 0) < len(t)]
            if dropped:
                notes.append("Cases with non-positive values (or values outside the domain of "
                             "the transformation) were excluded from: "
                             + ", ".join(MODELS[m] for m in dropped) + ".")
        if "logistic" in models:
            notes.append("Logistic upper bound: " + (f"{upper:g}" if upper else "infinity")
                         + ".")
        section.table(Table("Model Summary and Parameter Estimates", cols, rows,
                            footnotes=notes))
        if plot and fits:
            fig = figure(6.2, 4.4)
            ax = fig.add_subplot()
            ax.scatter(t, y, s=14, facecolors="none", edgecolors="#333333", lw=0.8,
                       label="Observed")
            grid = np.linspace(t.min(), t.max(), 300)
            for k, (m, res) in enumerate(fits):
                g = grid
                if m in ("logarithmic", "power"):
                    g = grid[grid > 0]
                if m in ("inverse", "s"):
                    g = grid[grid != 0]
                ax.plot(g, predict(m, res, g, upper), color=PALETTE[k % len(PALETTE)], lw=1.5,
                        label=MODELS[m])
            style(ax, dvar.title, ivar_title, "")
            ax.legend(fontsize=8, frameon=False)
            section.image(dvar.title, png(fig))
    return section


def _anova_table(section, dep_title, title, model, res, t, y, upper) -> None:
    X, resp, ok = _transform(model, t, y, upper)
    n = int(ok.sum())
    df1, df2 = res["df1"], res["df2"]
    sst = float(np.sum((resp[ok] - resp[ok].mean()) ** 2)) if res["constant"] else \
        float(np.sum(resp[ok] ** 2))
    sse = (1 - res["r2"]) * sst
    section.table(Table(f"{title} — ANOVA", ["", "Sum of Squares", "df", "Mean Square", "F",
                                            "Sig."], [
        ["Regression", fmt(sst - sse), fmt_int(df1), fmt((sst - sse) / df1), fmt(res["f"]),
         fmt_p(res["p"])],
        ["Residual", fmt(sse), fmt_int(df2), fmt(sse / df2) if df2 else "", "", ""],
        ["Total", fmt(sst), fmt_int(df1 + df2), "", "", ""]],
        footnotes=[f"Dependent Variable: {dep_title} (N = {n})"]))

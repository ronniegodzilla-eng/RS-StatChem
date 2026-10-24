"""Linear calibration curve for analytical chemistry (response y vs concentration x).

Equation, R², standard errors, LOD = 3.3·σ/b and LOQ = 10·σ/b (ICH Q2), with σ the residual
standard deviation s(y/x), the standard error of the intercept s(a) or a user-given blank SD,
and inverse prediction of unknown concentrations with confidence intervals following
Miller & Miller, *Statistics and Chemometrics for Analytical Chemistry* (sections 5.6, 5.10):

    s(x0) = s(y/x)/b · sqrt(1/(w0·m) + 1/n + (ȳ0 − ȳw)² / (b²·(Σwx² − n·x̄w²)))

Optional 1/x or 1/x² weighting uses weights normalised to Σw = n.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.analysis.common import (BAND, DATA, FIT, AnalysisResult, column, figure,
                                             intf, num, png, pval, scatter, style, var_title)
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table

SIGMAS = {"residual": "Residual standard deviation s(y/x)",
          "intercept": "Standard error of the intercept s(a)",
          "blank": "Standard deviation of the blank"}
CAL_WEIGHTS = {"none": "None", "1/x": "1/x", "1/x2": "1/x²"}


@dataclass
class Calibration:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    weighting: str
    slope: float
    intercept: float
    se_slope: float
    se_intercept: float
    s_yx: float
    xbar: float
    ybar: float
    sxx: float
    r: float
    r2: float
    rss: float
    tss: float
    sigma: float
    sigma_kind: str

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def dof(self) -> int:
        return self.n - 2

    @property
    def adj_r2(self) -> float:
        return 1 - (1 - self.r2) * (self.n - 1) / self.dof

    @property
    def lod(self) -> float:
        return 3.3 * self.sigma / abs(self.slope)

    @property
    def loq(self) -> float:
        return 10 * self.sigma / abs(self.slope)

    def predict(self, x) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, float)

    def weight_at(self, x0: float) -> float:
        if self.weighting == "none":
            return 1.0
        raw = _raw_weights(np.array([x0]), self.weighting)[0]
        return float(raw / np.mean(_raw_weights(self.x, self.weighting)))

    def inverse(self, y0: float, m: int = 1, level: float = 0.95) -> tuple[float, ...]:
        """(x0, s(x0), lower, upper) for a mean response ``y0`` of ``m`` replicates."""
        b = self.slope
        x0 = (y0 - self.intercept) / b
        w0 = self.weight_at(x0) if self.weighting != "none" and x0 > 0 else 1.0
        s = (self.s_yx / abs(b)) * math.sqrt(1 / (w0 * m) + 1 / self.n
                                             + (y0 - self.ybar) ** 2 / (b ** 2 * self.sxx))
        q = stats.t.ppf(0.5 + level / 2, self.dof)
        return x0, s, x0 - q * s, x0 + q * s

    def band(self, x, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, float)
        q = stats.t.ppf(0.5 + level / 2, self.dof)
        half = q * self.s_yx * np.sqrt(1 / self.n + (x - self.xbar) ** 2 / self.sxx)
        yhat = self.predict(x)
        return yhat - half, yhat + half


def _raw_weights(x, weighting) -> np.ndarray:
    x = np.asarray(x, float)
    if weighting == "1/x":
        return 1 / x
    if weighting == "1/x2":
        return 1 / x ** 2
    return np.ones(len(x))


def calibrate(x, y, weighting: str = "none", sigma: str = "residual",
              blank_sd: float | None = None) -> Calibration:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        raise ValueError("A calibration curve needs at least 3 standards.")
    if weighting != "none" and np.any(x <= 0):
        raise ValueError("Weighted calibration needs positive concentrations (remove the "
                         "blank / zero standard or use no weighting).")
    raw = _raw_weights(x, weighting)
    w = raw / raw.mean()
    xbar = float(np.sum(w * x) / n)
    ybar = float(np.sum(w * y) / n)
    sxx = float(np.sum(w * x * x) - n * xbar ** 2)
    if sxx <= 0:
        raise ValueError("The concentrations do not vary.")
    sxy = float(np.sum(w * x * y) - n * xbar * ybar)
    b = sxy / sxx
    a = ybar - b * xbar
    resid = y - (a + b * x)
    rss = float(np.sum(w * resid ** 2))
    tss = float(np.sum(w * (y - ybar) ** 2))
    s_yx = math.sqrt(rss / (n - 2))
    se_b = s_yx / math.sqrt(sxx)
    se_a = s_yx * math.sqrt(np.sum(w * x * x) / (n * sxx))
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    r = math.copysign(math.sqrt(max(r2, 0)), b)
    if b == 0:
        raise ValueError("The slope is zero; concentrations cannot be predicted.")
    if sigma == "residual":
        sig = s_yx
    elif sigma == "intercept":
        sig = se_a
    elif sigma == "blank":
        if not blank_sd or blank_sd <= 0:
            raise ValueError("Enter the standard deviation of the blank (> 0).")
        sig = float(blank_sd)
    else:
        raise ValueError(f"Unknown σ choice '{sigma}'")
    return Calibration(x, y, w, weighting, b, a, se_b, se_a, s_yx, xbar, ybar, sxx, r, r2, rss,
                       tss, sig, sigma)


def _lack_of_fit(cal: Calibration):
    keys, inv = np.unique(cal.x, return_inverse=True)
    k = len(keys)
    df_pe, df_lof = cal.n - k, k - 2
    if df_pe <= 0 or df_lof <= 0:
        return None
    sw = np.bincount(inv, cal.w)
    mean = np.bincount(inv, cal.w * cal.y) / sw
    ss_pe = float(np.sum(cal.w * (cal.y - mean[inv]) ** 2))
    ss_lof = max(cal.rss - ss_pe, 0.0)
    f = (ss_lof / df_lof) / (ss_pe / df_pe) if ss_pe > 0 else float("nan")
    return df_lof, ss_lof, df_pe, ss_pe, f, float(stats.f.sf(f, df_lof, df_pe))


def calibration_curve(ds: Dataset, xname: str, yname: str, unknown: str | None = None,
                      unknown_values=(), replicates: int = 1, sigma: str = "residual",
                      blank_sd: float | None = None, weighting: str = "none",
                      level: float = 0.95, save: bool = False) -> AnalysisResult:
    frame = pd.DataFrame({"x": column(ds, xname), "y": column(ds, yname)}).dropna()
    cal = calibrate(frame["x"], frame["y"], weighting, sigma, blank_sd)
    xt, yt = var_title(ds, xname), var_title(ds, yname)
    pc = f"{level * 100:g}%"
    section = Section("Calibration Curve")
    section.log(f"CALIBRATION X={xname} Y={yname}"
                + (f" UNKNOWN={unknown}" if unknown else "")
                + f" /SIGMA={sigma.upper()} /WEIGHT={weighting.upper()} /CI={level * 100:g}"
                + (f" /REPLICATES={replicates}" if replicates != 1 else "") + ".")
    sign = "+" if cal.intercept >= 0 else "−"
    eq = f"y = {num(cal.slope)}·x {sign} {num(abs(cal.intercept))}"
    section.table(Table("Calibration Equation", ["", "Value"], [
        ["Equation", eq], ["Concentration (X)", xt], ["Response (Y)", yt],
        ["Weighting", CAL_WEIGHTS[weighting]], ["Number of Standards", intf(cal.n)],
        ["Calibration Range", f"{num(cal.x.min())} – {num(cal.x.max())}"]]))
    q = stats.t.ppf(0.5 + level / 2, cal.dof)
    t_b, t_a = cal.slope / cal.se_slope, cal.intercept / cal.se_intercept
    rows = [["Intercept (a)", num(cal.intercept), num(cal.se_intercept), num(t_a, 5),
             pval(2 * stats.t.sf(abs(t_a), cal.dof)), num(cal.intercept - q * cal.se_intercept),
             num(cal.intercept + q * cal.se_intercept)],
            ["Slope (b, sensitivity)", num(cal.slope), num(cal.se_slope), num(t_b, 5),
             pval(2 * stats.t.sf(abs(t_b), cal.dof)), num(cal.slope - q * cal.se_slope),
             num(cal.slope + q * cal.se_slope)]]
    section.table(Table("Parameters", ["", "Value", "Standard Error", "t-Value", "Prob>|t|",
                                       (f"{pc} Confidence Interval", "LCL"),
                                       (f"{pc} Confidence Interval", "UCL")], rows))
    stat_rows = [["Number of Points", intf(cal.n)], ["Degrees of Freedom", intf(cal.dof)],
                 ["Pearson's r", num(cal.r)], ["R-Square (COD)", num(cal.r2)],
                 ["Adj. R-Square", num(cal.adj_r2)],
                 ["Residual Standard Deviation s(y/x)", num(cal.s_yx)],
                 ["Residual Sum of Squares", num(cal.rss)]]
    section.table(Table("Statistics", ["", yt], stat_rows))
    section.table(Table("Limits of Detection and Quantitation", ["", "Value"], [
        ["σ used", SIGMAS[sigma]], ["σ", num(cal.sigma)],
        ["Limit of Detection (LOD = 3.3σ/b)", num(cal.lod)],
        ["Limit of Quantitation (LOQ = 10σ/b)", num(cal.loq)]],
        footnotes=["LOD and LOQ are in concentration units (ICH Q2(R1), based on the "
                   "standard deviation of the response and the slope)."]))
    lof = _lack_of_fit(cal)
    msr = cal.tss - cal.rss
    f = msr / (cal.rss / cal.dof) if cal.rss > 0 else float("nan")
    arows = [["Model", "1", num(msr), num(msr), num(f), pval(stats.f.sf(f, 1, cal.dof))],
             ["Error", intf(cal.dof), num(cal.rss), num(cal.rss / cal.dof), "", ""]]
    if lof:
        arows += [["  Lack of Fit", intf(lof[0]), num(lof[1]), num(lof[1] / lof[0]),
                   num(lof[4]), pval(lof[5])],
                  ["  Pure Error", intf(lof[2]), num(lof[3]), num(lof[3] / lof[2]), "", ""]]
    arows.append(["Total", intf(cal.n - 1), num(cal.tss), "", "", ""])
    notes = []
    if lof:
        notes.append("A significant Lack of Fit (Prob>F < 0.05) indicates non-linearity.")
    section.table(Table("ANOVA", ["", "DF", "Sum of Squares", "Mean Square", "F Value",
                                  "Prob>F"], arows, footnotes=notes))

    unknown_rows, preds, saved_index = [], [], None
    values = [float(v) for v in unknown_values]
    labels = [f"Sample {i + 1}" for i in range(len(values))]
    if unknown:
        ser = column(ds, unknown).dropna()
        saved_index = ser.index
        values = list(ser.to_numpy()) + values
        labels = [f"Case {i + 1}" for i in ser.index] + labels
    lo_x, hi_x = cal.x.min(), cal.x.max()
    for label, y0 in zip(labels, values):
        x0, s, lo, hi = cal.inverse(y0, replicates, level)
        flag = []
        if x0 < cal.lod:
            flag.append("< LOD")
        elif x0 < cal.loq:
            flag.append("< LOQ")
        if x0 < lo_x or x0 > hi_x:
            flag.append("outside calibration range")
        unknown_rows.append([label, num(y0), num(x0), num(s), num(lo), num(hi), "; ".join(flag)])
        preds.append((x0, lo, hi))
    if unknown_rows:
        section.table(Table("Inverse Prediction of Unknown Concentrations", [
            "", "Response", "Concentration", "Standard Error",
            (f"{pc} Confidence Interval", "Lower"), (f"{pc} Confidence Interval", "Upper"),
            "Remark"], unknown_rows, footnotes=[
                f"Each response is the mean of m = {replicates} replicate measurement(s); "
                "s(x0) after Miller & Miller."]))
    section.image("Calibration Curve", _plot(cal, xt, yt, level, values, preds))
    section.image("Residual Plot", _residual_plot(cal, xt, yt), width=520)
    out = AnalysisResult(section)
    if save and unknown and saved_index is not None and len(saved_index):
        k = len(saved_index)
        arr = np.array(preds[:k])
        for key, col, text in (("CONC", 0, "Predicted concentration"),
                               ("CONCLO", 1, f"Lower {pc} CI of concentration"),
                               ("CONCHI", 2, f"Upper {pc} CI of concentration")):
            out.saved[key] = (f"{text} ({unknown})",
                              pd.Series(arr[:, col], index=saved_index).reindex(ds.df.index))
    return out


def _plot(cal: Calibration, xt, yt, level, unknowns, preds) -> bytes:
    fig = figure(6.2, 4.2)
    ax = fig.add_subplot()
    xs = [cal.x.min(), cal.x.max()] + [p[0] for p in preds]
    lo, hi = min(xs), max(xs)
    pad = 0.04 * (hi - lo or 1)
    xx = np.linspace(lo - pad, hi + pad, 300)
    a, b = cal.band(xx, level)
    ax.fill_between(xx, a, b, color=BAND, alpha=0.55, lw=0,
                    label=f"{level * 100:g}% Confidence Band")
    ax.plot(xx, cal.predict(xx), color=FIT, lw=1.8, label="Calibration Line")
    scatter(ax, cal.x, cal.y, label="Standards", color=DATA)
    if preds:
        px = [p[0] for p in preds]
        ax.errorbar(px, unknowns, xerr=[[p[0] - p[1] for p in preds], [p[2] - p[0]
                                                                         for p in preds]],
                    fmt="D", color="#1565C0", ms=5, capsize=3, lw=1, label="Unknowns")
    style(ax, "Calibration Curve", xt, yt)
    ax.text(0.02, 0.97, f"y = {num(cal.slope, 5)}x {'+' if cal.intercept >= 0 else '−'} "
            f"{num(abs(cal.intercept), 5)}\nR² = {cal.r2:.5f}\nLOD = {num(cal.lod, 4)}, "
            f"LOQ = {num(cal.loq, 4)}", transform=ax.transAxes, va="top", fontsize=8.5,
            bbox={"boxstyle": "round", "fc": "white", "ec": "#cccccc"})
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return png(fig)


def _residual_plot(cal: Calibration, xt, yt) -> bytes:
    fig = figure(6.2, 2.6)
    ax = fig.add_subplot()
    r = cal.y - cal.predict(cal.x)
    ax.axhline(0, color=FIT, lw=1)
    scatter(ax, cal.x, r, label="Residual")
    style(ax, "Residuals", xt, f"Residual of {yt}")
    return png(fig)
